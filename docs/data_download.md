# Data Download Guide

## Dataset: AfriSenti-SemEval 2023 — Amharic

## Loading the Dataset
The dataset loads automatically in the notebooks using the HuggingFace datasets library.
No manual download needed — just run the notebook cells.

```python
from datasets import load_dataset
dataset = load_dataset("masakhane/afrisenti", "amh")
```

### Dataset Structure
```
Splits:
  - train: 5,984 samples
  - validation: 1,497 samples
  - test: 1,999 samples

Labels:
  - positive
  - negative
  - neutral

Language: Amharic (Ethiopic script — ግዕዝ)
Domain: Twitter/social media
```

### Label Distribution

The training split contains approximately:

| Label | Percentage |
|---|---:|
| Neutral | 51.9% |
| Negative | 25.9% |
| Positive | 22.3% |

The test split contains:

| Label | Samples |
|---|---:|
| Negative | 1,337 |
| Positive | 438 |
| Neutral | 224 |
