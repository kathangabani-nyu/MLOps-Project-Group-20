#!/usr/bin/env python3
import os
import json
import argparse
import logging
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report

# must match your ONNX/class_map
CLASS_MAP = {
    0: "Central Serous Chorioretinopathy",
    1: "Diabetic Retinopathy",
    # … fill in all 15 classes …
    14: "Unused-Class-14"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def load_model(path, device):
    model = torch.load(path, map_location=device)
    model.to(device).eval()
    return model

def preprocess(image_path, device):
    pre = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return pre(img).unsqueeze(0).to(device)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    y_true, y_pred = [], []
    for entry in os.listdir(args.data_dir):
        if not entry.lower().endswith((".png",".jpg")):
            continue
        path = os.path.join(args.data_dir, entry)
        label = int(entry.split("_")[0])  # assume filename like "0_img1.png"
        inp = preprocess(path, device)
        with torch.no_grad():
            logits = model(inp)
        pred = logits.argmax(dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

    report = classification_report(
        y_true, y_pred, target_names=[CLASS_MAP[i] for i in sorted(CLASS_MAP)]
    )
    logging.info("Evaluation Report:\n%s", report)

    # save metrics
    out = {
        "accuracy": np.mean(np.array(y_true) == np.array(y_pred)).item()
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logging.info("Saved metrics to %s", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default=os.getenv("TRAINED_MODEL_PATH","./model.pth"))
    p.add_argument("--data-dir",
                   dest="data_dir",
                   default=os.getenv("TEST_DATA_DIR","./test_images"))
    p.add_argument("--output",
                   default=os.getenv("METRICS_OUTPUT","metrics.json"))
    args = p.parse_args()
    main(args)
