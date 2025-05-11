import streamlit as st
import pandas as pd
import json
from pathlib import Path
from collections import Counter
import plotly.express as px
from itertools import islice

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Business Insights Dashboard", layout="wide")
st.title("Business Insights Dashboard")
st.markdown(
    "Visualizations generated from your train/validation/test JSONL files in\n"
    "`/mnt/block/MLOps-Project-Group-20/data`."
)

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = Path("/mnt/block") / "MLOps-Project-Group-20" / "data"

# ── JSONL loader with optional sampling ───────────────────────────────────────
@st.cache_data
def read_jsonl(filename: str, max_lines: int = None) -> pd.DataFrame:
    file_path = DATA_DIR / filename
    if not file_path.exists():
        return pd.DataFrame()
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)

# ── Load each split (sample train for speed) ──────────────────────────────────
TRAIN_SAMPLE_SIZE = 20_000
train_df      = read_jsonl("train.jsonl",      max_lines=TRAIN_SAMPLE_SIZE)
validation_df = read_jsonl("validation.jsonl")  # full, small enough
test_df       = read_jsonl("test.jsonl")        # full, small enough

# ── 1) Sample Counts ──────────────────────────────────────────────────────────
st.header("Sample Counts")
counts = {
    "Train":      len(train_df),
    "Validation": len(validation_df),
    "Test":       len(test_df),
}
counts = {k: v for k, v in counts.items() if v > 0}
fig_counts = px.bar(
    x=list(counts.keys()),
    y=list(counts.values()),
    labels={"x": "Split", "y": "Number of Samples"},
    title="Number of Samples in Each Split"
)
st.plotly_chart(fig_counts, use_container_width=True)

# ── 2) Article Length Distribution ────────────────────────────────────────────
st.header("Article Length Distribution (Words)")
for name, df in [("Train", train_df), ("Test", test_df)]:
    if not df.empty and "article" in df.columns:
        wc = df["article"].dropna().astype(str).apply(lambda txt: len(txt.split()))
        fig_wc = px.histogram(
            x=wc,
            nbins=50,
            labels={"x": "Article Word Count", "y": "Number of Articles"},
            title=f"{name} Articles: Word Count Distribution"
        )
        st.plotly_chart(fig_wc, use_container_width=True)

# ── 3) Split-based Exploration ─────────────────────────────────────────────────
# Sidebar selector (now here so it doesn’t suppress the above)
available = [s for s in ("train", "validation", "test") if (DATA_DIR / f"{s}.jsonl").exists()]
choice = st.sidebar.selectbox("Select dataset split", [s.capitalize() for s in available])
split  = choice.lower()
df     = read_jsonl(f"{split}.jsonl", max_lines=(TRAIN_SAMPLE_SIZE if split == "train" else None))

# a) Overview
st.header(f"{choice} Set Overview")
st.write(f"**Rows:** {df.shape[0]} **Columns:** {df.shape[1]}")
if not df.empty:
    st.dataframe(df.head(), height=300)

# b) Label distribution
if "label" in df.columns:
    st.subheader(f"{choice} Label Distribution")
    lc = df["label"].value_counts()
    fig_lbl = px.bar(
        x=lc.index.astype(str),
        y=lc.values,
        labels={"x": "Label", "y": "Count"},
        title=f"{choice} Label Distribution"
    )
    st.plotly_chart(fig_lbl, use_container_width=True)

# c) Top-20 Words in 'article'
if "article" in df.columns:
    st.subheader(f"Top 20 Words in {choice} Articles")
    ctr = Counter()
    for text in df["article"].dropna().astype(str):
        ctr.update(text.lower().split())
    words, counts = zip(*ctr.most_common(20)) if ctr else ([], [])
    fig_words = px.bar(
        x=list(words),
        y=list(counts),
        labels={"x": "Word", "y": "Count"},
        title=f"{choice} Articles: Top 20 Words"
    )
    st.plotly_chart(fig_words, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Dashboard running on **kvm@tacc**, reading live from block storage.")
