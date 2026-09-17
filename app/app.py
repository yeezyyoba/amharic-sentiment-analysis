"""
app/app.py
----------
Streamlit web app for Amharic Sentiment Analysis.
Powered by fine-tuned Afro-XLM-R on AfriSenti Amharic dataset.
"""

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Amharic Sentiment Analyzer",
    page_icon="🇪🇹",
    layout="centered"
)

# ── Constants ──────────────────────────────────────────────────
MODEL_PATH = "yeezyyoba/amharic-sentiment-transformer"
MAX_LENGTH = 72
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_EMOJI = {"positive": "😊 Positive", "negative": "😠 Negative", "neutral": "😐 Neutral"}
LABEL_COLOR = {"positive": "#1D9E75", "negative": "#E24B4A", "neutral": "#534AB7"}

# ── Example tweets ─────────────────────────────────────────────
EXAMPLES = {
    "positive": "መልካም አዲስ አመት! እግዚአብሄር ይባርካችሁ።",
    "negative": "ይህ መንግስት ህዝቡን እየጨቆነ ነው። ምንም ለውጥ የለም።",
    "neutral":  "ጠቅላይ ሚኒስቴሩ ዛሬ ጠዋት ስብሰባ አካሂደዋል።"
}

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

# ── Predict ────────────────────────────────────────────────────
def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id = np.argmax(probs)
    return ID2LABEL[pred_id], probs

# ── UI ─────────────────────────────────────────────────────────
st.title("🇪🇹 Amharic Sentiment Analyzer")
st.markdown(
    "Powered by **Afro-XLM-R** fine-tuned on [AfriSenti](https://huggingface.co/datasets/masakhane/afrisenti) · "
    "Macro F1: **0.5543 ± 0.0068**"
)
st.divider()

# Load model
with st.spinner("Loading model..."):
    try:
        tokenizer, model = load_model()
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure the Hugging Face model repository is accessible.")
        st.stop()

# Example buttons
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

if col1.button("😊 Positive example"):
    st.session_state["input_text"] = EXAMPLES["positive"]
if col2.button("😠 Negative example"):
    st.session_state["input_text"] = EXAMPLES["negative"]
if col3.button("😐 Neutral example"):
    st.session_state["input_text"] = EXAMPLES["neutral"]

# Text input
text_input = st.text_area(
    "Enter Amharic text:",
    value=st.session_state.get("input_text", ""),
    height=120,
    placeholder="ጽሑፍዎን እዚህ ያስገቡ..."
)

# Predict button
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some Amharic text.")
    else:
        with st.spinner("Analyzing..."):
            label, probs = predict(text_input, tokenizer, model)

        # Result
        st.divider()
        color = LABEL_COLOR[label]
        emoji_label = LABEL_EMOJI[label]

        st.markdown(f"### Prediction: <span style='color:{color}'>{emoji_label}</span>",
                    unsafe_allow_html=True)

        # Confidence bars
        st.markdown("**Confidence scores:**")
        for i, (lbl, prob) in enumerate(zip(["negative", "neutral", "positive"], probs)):
            col_a, col_b = st.columns([1, 4])
            col_a.markdown(f"**{LABEL_EMOJI[lbl]}**")
            col_b.progress(float(prob), text=f"{prob*100:.1f}%")

        # Bar chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        classes = ["Negative", "Neutral", "Positive"]
        colors  = [LABEL_COLOR["negative"], LABEL_COLOR["neutral"], LABEL_COLOR["positive"]]
        bars = ax.bar(classes, probs, color=colors, alpha=0.85)
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{prob*100:.1f}%", ha="center", fontsize=10)
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Sentiment Confidence Scores", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()

# About section
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model**: Davlan/afro-xlmr-base (fine-tuned)  
    **Dataset**: AfriSenti-SemEval 2023 — Amharic subset  
    **Training**: 3 seeds (42, 123, 456) for statistical reliability  
    **Result**: Macro F1 = 0.5543 ± 0.0068  
    **Baseline**: TF-IDF + Logistic Regression (Macro F1 = 0.4640)  
    **Improvement**: +9.03 points over baseline  

    **Known limitations**:
    - News-framed negative content often predicted as neutral
    - Constructive criticism may be misclassified as neutral
    - Sarcasm and irony remain challenging
    """)

with st.expander("📄 Research"):
    st.markdown("""
    This app is part of a research project on Amharic NLP:

    *"Improving Amharic Sentiment Analysis through Ethiopic Character 
    Normalization and Multilingual Transformer Fine-tuning"*

    **Author**: Eyob Nebyou, Addis Ababa University  
    **GitHub**: [amharic-sentiment-analysis](https://github.com/yeezyyoba/amharic-sentiment-analysis)
    """)
