import streamlit as st
import pandas as pd
import json
from pathlib import Path
from collections import Counter
import plotly.express as px
from itertools import islice

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Business Insights Dashboard", layout="wide")
st.title("Business Insights Dashboard")
st.markdown(
    "Visualizations generated from your train/validation/test JSONL files in\n"
    "`/mnt/block/MLOps-Project-Group-20/data`."
)

# ── Data directory ───────────────────────────────────────────────────────────
DATA_DIR = Path("/mnt/block") / "MLOps-Project-Group-20" / "data"

# ── JSONL loader with optional sampling ──────────────────────────────────────
@st.cache_data
def read_jsonl(filename: str, max_lines: int = None) -> pd.DataFrame:
    """Load up to max_lines (or entire file if None) from JSONL into DataFrame."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        st.error(f"Data file not found: {file_path}")
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

# ── Determine which splits are present ────────────────────────────────────────
splits = ["train", "validation", "test"]
available = []
for s in splits:
    if (DATA_DIR / f"{s}.jsonl").exists():
        available.append(s)
if not available:
    st.error("No JSONL files found in data directory.")
    st.stop()

# ── Sidebar selector ─────────────────────────────────────────────────────────
choice = st.sidebar.selectbox(
    "Select dataset split", [s.capitalize() for s in available]
)
split = choice.lower()

# ── Load the DataFrame (sample train for speed) ───────────────────────────────
if split == "train":
    df = read_jsonl("train.jsonl", max_lines=100_000)
else:
    df = read_jsonl(f"{split}.jsonl", max_lines=None)

# ── Overview ─────────────────────────────────────────────────────────────────
st.header(f"{choice} Set Overview")
st.write(f"**Rows:** {df.shape[0]} **Columns:** {df.shape[1]}")
if not df.empty:
    st.dataframe(df.head(), height=300)

# ── Label distribution ───────────────────────────────────────────────────────
if "label" in df.columns:
    st.subheader("Label Distribution")
    lc = df["label"].value_counts()
    fig = px.bar(
        x=lc.index.astype(str),
        y=lc.values,
        labels={"x": "Label", "y": "Count"},
        title=f"{choice} Label Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Top 20 Words in 'article' ────────────────────────────────────────────────
if "article" in df.columns:
    st.subheader(f"Top 20 Words in {choice} Articles")
    # Flatten tokens and count
    counter = Counter()
    for text in df["article"].dropna().astype(str):
        counter.update(text.lower().split())
    words, counts = zip(*counter.most_common(20)) if counter else ([], [])
    fig2 = px.bar(
        x=list(words),
        y=list(counts),
        labels={"x": "Word", "y": "Count"},
        title=f"{choice} Articles: Top 20 Words"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Dashboard running on **kvm@tacc**, reading live from block storage.")
