# Amharic Sentiment Analysis with Afro-XLM-R

A research-oriented study of sentiment classification for Amharic social-media text using the AfriSenti Amharic dataset, Ethiopic-script normalization, classical TF-IDF baselines, and fine-tuning of Afro-XLM-R.

## 🚀 Live Demo

**Try the model:** [Amharic Sentiment Analysis — Live Demo](https://amharic-sentiment-analysis.streamlit.app/)

Enter Amharic text and see the model's predicted sentiment and class probabilities.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/docs/transformers)
[![Dataset](https://img.shields.io/badge/Dataset-AfriSenti-green?style=flat-square)](https://huggingface.co/datasets/masakhane/afrisenti)

---

## Overview

Amharic is a comparatively under-resourced language in natural language processing, with fewer publicly available resources and models than high-resource languages.

This project investigates **three-class sentiment classification for Amharic text** using the Amharic subset of the **AfriSenti** dataset.

The project follows a complete experimental pipeline:

```text
AfriSenti Amharic
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Ethiopic Text Normalization
       │
       ▼
TF-IDF Classical Baselines
       │
       ▼
Afro-XLM-R Fine-Tuning
       │
       ▼
Multi-Seed Evaluation
       │
       ▼
Error Analysis
       │
       ▼
Attention Visualization
```

The goal is not only to measure model performance, but also to investigate **where lexical baselines and multilingual transformer models succeed or fail on Amharic text**.

---

## Research Questions

The project investigates the following questions:

1. What are the characteristics and class distribution of the AfriSenti Amharic dataset?
2. How does Ethiopic-script normalization affect the text used for classification?
3. How well can a TF-IDF + Logistic Regression model classify Amharic sentiment?
4. How does fine-tuned Afro-XLM-R compare with the classical baseline?
5. How consistent is transformer performance across multiple random seeds?
6. What types of Amharic examples remain difficult for the models?
7. What linguistic or contextual patterns appear in model errors?

---

## Dataset

The experiments use the **Amharic (`amh`) configuration of the AfriSenti dataset** from Hugging Face.

```python
from datasets import load_dataset

dataset = load_dataset(
    "masakhane/afrisenti",
    "amh"
)
```

The dataset contains three sentiment classes:

* `negative`
* `neutral`
* `positive`

### Dataset Splits

| Split      |   Samples |
| ---------- | --------: |
| Train      |     5,984 |
| Validation |     1,497 |
| Test       |     1,999 |
| **Total**  | **9,480** |

The training set contains:

* Neutral: 3,104
* Negative: 1,548
* Positive: 1,332

This class distribution motivates the use of **Macro-F1** as the primary evaluation metric.

---

## Methodology

### 1. Exploratory Data Analysis

The EDA stage examines:

* Dataset size and split structure
* Sentiment class distribution
* Tweet length distributions
* Common vocabulary by sentiment
* Characteristics of Amharic social-media text

See:

`notebooks/01_EDA.ipynb`

---

### 2. Ethiopic Text Normalization

The preprocessing pipeline applies Amharic-specific text normalization rather than relying exclusively on generic English-oriented preprocessing.

The normalization handles variations within Ethiopic character families while preserving the linguistic content needed for sentiment classification.

The preprocessing stage also handles:

* Unicode normalization
* Whitespace normalization
* URL and user-mention handling
* Duplicate removal
* Punctuation and symbol handling
* Text cleaning

See:

`notebooks/02_preprocessing.ipynb`

> **Preprocessing consideration:** Removing punctuation and special symbols can also remove sentiment-bearing information such as repeated punctuation or emojis. This is treated as a methodological limitation rather than assuming that these features are irrelevant.

---

### 3. Classical Baselines

A TF-IDF representation using unigram and bigram features is used to establish classical machine-learning baselines.

The evaluated models include:

* Logistic Regression
* Linear SVM
* Random Forest

The primary classical baseline is **TF-IDF + Logistic Regression**.

The vectorizer is fitted on the training data and then applied to validation and test data to avoid fitting preprocessing parameters on held-out examples.

See:

`notebooks/03_baseline_model.ipynb`

---

### 4. Transformer Fine-Tuning

The transformer experiment fine-tunes:

**`Davlan/afro-xlmr-base`**

Afro-XLM-R is a multilingual transformer model developed for African languages.

The main experimental configuration includes:

| Parameter               | Value                            |
| ----------------------- | -------------------------------- |
| Model                   | `Davlan/afro-xlmr-base`          |
| Maximum sequence length | 72                               |
| Epochs                  | 5                                |
| Batch size              | 32                               |
| Learning rate           | 2e-5                             |
| Random seeds            | 42, 123, 456                     |
| Task                    | 3-class sentiment classification |

The model is evaluated across multiple random seeds rather than relying exclusively on a single training run.

See:

`notebooks/06_transformer_finetuning.ipynb`

---

## Results

The primary evaluation metric is **Macro-F1**, which gives equal weight to each sentiment class and is useful when class frequencies are not balanced.

### Classical Baselines

| Model               | Validation Macro-F1 | Test Macro-F1 | Test Accuracy |
| ------------------- | ------------------: | ------------: | ------------: |
| Logistic Regression |              0.5256 |    **0.4640** |        0.4997 |
| Linear SVM          |              0.4972 |        0.4139 |        0.4277 |
| Random Forest       |              0.4975 |        0.4099 |        0.4202 |

### Transformer

| System                       |            Macro-F1 |
| ---------------------------- | ------------------: |
| TF-IDF + Logistic Regression |              0.4640 |
| **Afro-XLM-R Base**          | **0.5543 ± 0.0068** |

### Multi-Seed Results

|            Seed |            Macro-F1 |
| --------------: | ------------------: |
|              42 |              0.5623 |
|             123 |              0.5553 |
|             456 |              0.5455 |
| **Mean ± Std.** | **0.5543 ± 0.0068** |

The multi-seed transformer result is **0.0903 Macro-F1 points higher** than the TF-IDF + Logistic Regression baseline.

The individual seed-42 result is reported separately from the three-seed mean to avoid conflating a single run with the overall experiment.

See:

`notebooks/07_evaluation.ipynb`

---

## Error Analysis

Performance metrics alone do not explain why sentiment classification is difficult for Amharic text.

The project therefore includes qualitative error analysis comparing:

* Examples correctly classified by both models
* Examples correctly classified only by the transformer
* Examples correctly classified only by the TF-IDF baseline
* Examples misclassified by both models

The analysis examines patterns including:

* Negation
* Context-dependent sentiment
* Journalistic/news framing
* Repetition
* Culturally specific expressions
* Ambiguous sentiment
* Noisy social-media language

The purpose of the analysis is to identify **patterns in model behavior**, rather than treating individual predictions as direct evidence of linguistic understanding.

See:

`notebooks/08_error_analysis.ipynb`

---

## Attention Visualization

The project also explores transformer attention patterns for selected Amharic examples.

These visualizations are used as an interpretive aid to investigate which tokens receive comparatively high attention in particular examples.

Attention visualizations should not be interpreted as a complete explanation of model reasoning.

---

## Interactive Demo

A Streamlit application is included to demonstrate sentiment predictions using the fine-tuned transformer.

### Live Application

**[Launch the Amharic Sentiment Analysis Demo](https://amharic-sentiment-analysis.streamlit.app/)**

The application provides:

* Amharic text input
* Predicted sentiment
* Class probabilities
* Example inputs
* Model information

The application is intended as a demonstration interface for the research model rather than as a separate production system.

### Run Locally

```bash
streamlit run app/app.py
```

---

## Project Structure

```text
amharic-sentiment-analysis/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── data_download.md
│   └── final_report.md
│
├── models/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_*.ipynb
│   ├── 05_*.ipynb
│   ├── 06_transformer_finetuning.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_error_analysis.ipynb
│
├── reports/
│
├── src/
│   └── data/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Reproducibility

### 1. Clone the Repository

```bash
git clone https://github.com/yeezyyoba/amharic-sentiment-analysis.git
cd amharic-sentiment-analysis
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venvScriptsactivate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

The notebooks are organized as a sequential research workflow:

```text
01 → EDA
02 → preprocessing
03 → classical baselines
04 → ...
05 → ...
06 → transformer fine-tuning
07 → evaluation
08 → error analysis
```

The transformer fine-tuning experiment requires a suitable GPU environment for practical training times.

---

## Limitations

Several limitations remain:

* The dataset is relatively small compared with datasets available for high-resource languages.
* The dataset contains only three sentiment classes.
* Social-media text contains spelling variation, slang, code-switching, and noisy expressions.
* Normalization may remove information that is useful for sentiment classification.
* The current transformer experiment uses the **base** Afro-XLM-R model.
* The project does not perform extensive hyperparameter optimization.
* Qualitative error analysis is based on selected examples and should not be interpreted as a statistically complete linguistic analysis.
* Attention visualizations provide interpretive signals but do not constitute a complete explanation of model decisions.

---

## Future Work

Potential extensions include:

* Fine-tuning `afro-xlmr-large`
* More extensive hyperparameter search
* Emoji- and punctuation-aware preprocessing
* Amharic-specific data augmentation
* Better handling of code-switching
* Larger Amharic sentiment datasets
* Comparison with additional multilingual and Amharic-specific language models
* More systematic linguistic error categorization
* Calibration and uncertainty analysis
* Human evaluation of difficult and ambiguous examples

---

## Author

**Eyob Nebyou**

BSc Computer Science
Addis Ababa University

GitHub: [@yeezyyoba](https://github.com/yeezyyoba)

---

## License

This project is released under the MIT License.
