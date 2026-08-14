# Amharic Sentiment Analysis Platform
### Low-Resource NLP with Multilingual Transformers | End-to-End NLP Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

An end-to-end NLP project that fine-tunes a multilingual transformer (Afro-XLM-R) on Amharic social media text to classify sentiment as **positive**, **negative**, or **neutral**. Amharic is a low-resource language spoken by 50+ million people in Ethiopia — making this a meaningful contribution to African NLP research.

The project compares a classical TF-IDF + SVM baseline against a state-of-the-art transformer, with a live deployed web application for real-time inference.

---

## Problem Statement

Sentiment analysis tools exist for English, French, Arabic — but almost nothing exists for Amharic. Ethiopian businesses, researchers, and policymakers have no automated way to understand public opinion expressed in the national language. This project builds that tool.

---

## Dataset

**AfriSenti-SemEval 2023 — Amharic Subset**
- Source: [HuggingFace — shmuhammad/AfriSenti-twitter-sentiment](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment)
- Language: Amharic (am)
- Labels: Positive, Negative, Neutral
- Domain: Twitter/social media text

---

## Project Structure

```
amharic-sentiment-analysis/
│
├── data/
│   ├── raw/              # Original AfriSenti dataset files
│   ├── processed/        # Cleaned and tokenized data
│   └── external/         # Reference data
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # Dataset exploration
│   ├── 02_preprocessing.ipynb          # Text cleaning pipeline
│   ├── 03_baseline_model.ipynb         # TF-IDF + SVM baseline
│   ├── 04_transformer_finetuning.ipynb # Afro-XLM-R fine-tuning
│   ├── 05_evaluation.ipynb             # Model comparison
│   └── 06_error_analysis.ipynb         # Error & attention analysis
│
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model training utilities
│   └── visualization/    # Plotting functions
│
├── app/
│   └── app.py            # Streamlit web application
│
├── models/               # Saved model checkpoints
├── reports/              # Charts, figures, final report
├── docs/                 # Model card, documentation
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Classical NLP | scikit-learn, TF-IDF, SVM |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Model | Afro-XLM-R (multilingual transformer) |
| Deployment | Streamlit, HuggingFace Spaces |
| Visualization | matplotlib, seaborn |

---

## Results

| Model | Macro F1 | Accuracy |
|---|---|---|
| TF-IDF + SVM (baseline) | TBD | TBD |
| Afro-XLM-R (fine-tuned) | TBD | TBD |

*Results will be updated as training progresses.*

---

## Live Demo

🚀 **[Try the live app here](#)** *(link will be added on Day 11)*

---

## Weekly Progress

- [x] Day 1 — Project setup & repo structure
- [ ] Day 2 — Dataset EDA
- [ ] Day 3 — Text preprocessing pipeline
- [ ] Day 4 — TF-IDF + SVM baseline
- [ ] Day 5 — Baseline error analysis
- [ ] Day 6 — HuggingFace tokenization
- [ ] Day 7 — Afro-XLM-R fine-tuning
- [ ] Day 8 — Model evaluation & comparison
- [ ] Day 9 — Error analysis & attention viz
- [ ] Day 10 — Streamlit app
- [ ] Day 11 — HuggingFace Spaces deployment
- [ ] Day 12 — Model card & documentation
- [ ] Day 13 — Final report & README polish
- [ ] Day 14 — Final review & v1.0 release

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/amharic-sentiment-analysis.git
cd amharic-sentiment-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/app.py
```

---

## Author

**Eyob Nebyou**
Computer Science Student, Addis Ababa University
[LinkedIn](https://linkedin.com/in/eyob-nebyou-2782b8395) | [GitHub](https://github.com/yeezyyoba)

---

## License

MIT License
