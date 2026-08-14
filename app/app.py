"""
app/app.py
----------
Streamlit web app for Amharic Sentiment Analysis.
Full implementation will be added on Day 10.
"""

import streamlit as st

st.set_page_config(
    page_title="Amharic Sentiment Analysis",
    page_icon="🇪🇹",
    layout="centered"
)

st.title("🇪🇹 Amharic Sentiment Analyzer")
st.markdown("*Powered by Afro-XLM-R — Fine-tuned on AfriSenti*")
st.info("Full app coming on Day 10. Model training in progress.")
