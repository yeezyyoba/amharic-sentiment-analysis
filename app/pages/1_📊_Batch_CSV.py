"""
app/pages/1_📊_Batch_CSV.py
---------------------------
Batch scoring page: upload a CSV, pick the text column, score every row
with the fine-tuned Afro-XLM-R model, and download the result.
"""

import io
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Batch CSV — Amharic Sentiment",
    page_icon="📊",
    layout="centered"
)

# ── Constants (match app.py) ───────────────────────────────────
MODEL_PATH = "yeezyyoba/amharic-sentiment-transformer"
MAX_LENGTH = 72
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_EMOJI = {"positive": "😊 Positive", "negative": "😠 Negative", "neutral": "😐 Neutral"}
LABEL_COLOR = {"positive": "#1D9E75", "negative": "#E24B4A", "neutral": "#534AB7"}
MAX_ROWS = 20_000

# ── Load model (same cache key as app.py, so it's shared, not reloaded) ──
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def predict_batch(texts, tokenizer, model, batch_size=32, on_progress=None):
    """Run the model over a list of strings in batches. Returns an (n, 3) array."""
    all_probs = []
    total = len(texts)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            chunk = texts[start:start + batch_size]
            inputs = tokenizer(
                chunk,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()
            all_probs.append(probs)
            if on_progress:
                on_progress(min(start + batch_size, total), total)
    return np.vstack(all_probs) if all_probs else np.zeros((0, 3))


def read_table(upload):
    """Read CSV/TSV/Excel with encoding fallbacks for Ethiopic text."""
    name = upload.name.lower()
    data = upload.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding, sep=None, engine="python")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read the file: {last_error}")


# ── UI ─────────────────────────────────────────────────────────
st.title("📊 Batch CSV Scoring")
st.markdown(
    "Upload a CSV, TSV, or Excel file. Every original column is kept, and "
    "`sentiment`, `confidence`, and per-class probability columns are appended."
)
st.divider()

with st.spinner("Loading model..."):
    try:
        tokenizer, model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

upload = st.file_uploader("Choose a file", type=["csv", "tsv", "txt", "xlsx", "xls"])

if upload is not None:
    try:
        df = read_table(upload)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    st.dataframe(df.head(5), use_container_width=True)

    guess = next(
        (c for c in df.columns if str(c).lower() in {"text", "tweet", "content", "message", "comment"}),
        df.columns[0]
    )
    text_col = st.selectbox("Which column holds the Amharic text?", list(df.columns),
                             index=list(df.columns).index(guess))
    batch_size = st.select_slider("Batch size", [16, 32, 64, 128], value=32)

    if len(df) > MAX_ROWS:
        st.warning(f"Only the first {MAX_ROWS:,} rows will be scored.")
        df = df.head(MAX_ROWS)

    if st.button("🔍 Score file", type="primary", use_container_width=True):
        texts = df[text_col].astype(str).fillna("")
        usable_mask = texts.str.strip().str.len().gt(0)
        note = pd.Series("", index=df.index, dtype=object)
        note[~usable_mask] = "empty row — skipped"

        to_score = texts[usable_mask].tolist()
        bar = st.progress(0.0, text="Scoring...")
        started = time.time()

        def tick(done, total):
            bar.progress(done / total, text=f"Scoring {done:,} / {total:,} rows")

        probs = predict_batch(to_score, tokenizer, model, batch_size=batch_size, on_progress=tick)
        bar.empty()

        pred_ids = np.argmax(probs, axis=1)
        result_cols = pd.DataFrame({
            "sentiment": [ID2LABEL[i] for i in pred_ids],
            "confidence": probs[np.arange(len(probs)), pred_ids].round(4),
            "p_negative": probs[:, 0].round(4),
            "p_neutral": probs[:, 1].round(4),
            "p_positive": probs[:, 2].round(4),
        }, index=texts[usable_mask].index)

        result = df.join(result_cols)
        result["note"] = note

        elapsed = time.time() - started
        st.success(f"Scored {len(to_score):,} rows in {elapsed:,.1f}s "
                   f"({(~usable_mask).sum():,} skipped).")
        st.session_state["batch_result"] = result

# ── Results dashboard ────────────────────────────────────────────
if "batch_result" in st.session_state:
    result = st.session_state["batch_result"]
    scored = result.dropna(subset=["sentiment"])

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(scored):,}")
    c2.metric("Mean confidence", f"{scored['confidence'].mean()*100:.1f}%")
    c3.metric("Low confidence (<55%)", f"{(scored['confidence'] < 0.55).sum():,}")

    st.markdown("**Label distribution**")
    counts = scored["sentiment"].value_counts()
    cols = st.columns(3)
    for i, label in enumerate(["negative", "neutral", "positive"]):
        n = int(counts.get(label, 0))
        cols[i].markdown(
            f"<div style='text-align:center'><span style='color:{LABEL_COLOR[label]}; "
            f"font-size:1.4rem; font-weight:bold'>{n}</span><br>{LABEL_EMOJI[label]}</div>",
            unsafe_allow_html=True
        )

    st.markdown("**20 least confident rows** — a good queue for manual review.")
    st.dataframe(scored.nsmallest(20, "confidence"), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download scored CSV",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="amharic_sentiment_scored.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )
