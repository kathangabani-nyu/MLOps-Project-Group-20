#!/usr/bin/env python3
import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def send_request(endpoint, image_path):
    start = time.time()
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            r = requests.post(endpoint, files=files, timeout=5)
        r.raise_for_status()
        latency = time.time() - start
        return latency, True
    except Exception as e:
        logging.warning(f"Request failed: {e}")
        return None, False

def main():
    endpoint = os.getenv("PREDICT_ENDPOINT", "http://localhost:3500/predict")
    images_dir = os.getenv("TEST_IMAGES_DIR", "./test_images")
    total = int(os.getenv("NUM_REQUESTS", "100"))
    concurrency = int(os.getenv("CONCURRENCY", "10"))

    # collect sample images
    images = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images:
        logging.error("No images found in %s", images_dir)
        exit(1)

    successes = 0
    latencies = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, endpoint, images[i % len(images)])
            for i in range(total)
        ]
        for fut in futures:
            lat, ok = fut.result()
            if ok:
                successes += 1
                latencies.append(lat)

    logging.info("Load test: %d/%d succeeded", successes, total)
    if latencies:
        avg = sum(latencies) / len(latencies)
        logging.info("Avg latency: %.3fs", avg)
    else:
        logging.error("All requests failed")

if __name__ == "__main__":
    main()
