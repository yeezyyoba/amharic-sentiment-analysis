# Data Download Guide

## Dataset: AfriSenti-SemEval 2023 — Amharic

### Option 1: HuggingFace Datasets (Recommended)
The dataset loads automatically in the notebooks using the HuggingFace datasets library.
No manual download needed — just run the notebook cells.

```python
from datasets import load_dataset
dataset = load_dataset("shmuhammad/AfriSenti-twitter-sentiment", "amh")
```

### Option 2: Manual Download
- URL: https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment
- Click "Files and versions" tab
- Download the `amh/` folder files
- Place in `data/raw/amharic/`

### Dataset Structure
```
Splits:
  - train: ~5,985 samples
  - validation: ~595 samples  
  - test: ~2,000 samples

Labels:
  - positive
  - negative
  - neutral

Language: Amharic (Ethiopic script — ግዕዝ)
Domain: Twitter/social media
```

### Label Distribution (approximate)
| Label | Train | Validation | Test |
|---|---|---|---|
| Positive | ~45% | ~45% | ~45% |
| Negative | ~30% | ~30% | ~30% |
| Neutral | ~25% | ~25% | ~25% |
