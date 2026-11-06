"""Regenerate the three paper figures from saved models and data.

Run from the repository root with the project venv:
    venv/bin/python scripts/make_paper_figures.py

Outputs (in reports/):
    paper_fig1_label_shift.png
    paper_fig2_confusion.png
    paper_fig3_f1.png
    test_predictions_seed42.csv   (gold, baseline and seed-42 transformer predictions)
    paper_figures_check.json      (confusion counts and recomputed scores, for checking)
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["negative", "neutral", "positive"]
NAMES = ["Negative", "Neutral", "Positive"]
COLORS = {"negative": "#E05A5A", "neutral": "#6A64C0", "positive": "#3FAA87"}
MAX_LENGTH = 72
OUT = "reports"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ── Figure 1: label distribution per split ─────────────────────
splits = [("Train", "train_raw"), ("Validation", "val_raw"), ("Test", "test_raw")]
pct = {}
for name, f in splits:
    d = pd.read_parquet(f"data/processed/{f}.parquet")
    vc = d["sentiment"].value_counts()
    pct[name] = [100 * vc.get(l, 0) / len(d) for l in LABELS]

fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(splits))
w = 0.26
for i, l in enumerate(LABELS):
    vals = [pct[s][i] for s, _ in splits]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=NAMES[i], color=COLORS[l])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(x, [s for s, _ in splits])
ax.set_ylabel("% of split")
ax.set_ylim(0, 75)
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/paper_fig1_label_shift.png", dpi=300)
plt.close(fig)

# ── Predictions: baseline and seed-42 transformer ──────────────
test = pd.read_parquet("data/processed/test_clean.parquet")
y = test["sentiment"].tolist()

clf = joblib.load("models/baseline_model.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
base_pred = list(clf.predict(tfidf.transform(test["clean_tweet"])))

tok = AutoTokenizer.from_pretrained("models/best_transformer")
model = AutoModelForSequenceClassification.from_pretrained("models/best_transformer").eval()
id2label = model.config.id2label
tr_pred = []
texts = test["clean_tweet"].tolist()
with torch.no_grad():
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, padding="max_length",
                  max_length=MAX_LENGTH, return_tensors="pt")
        ids = model(**enc).logits.argmax(-1).tolist()
        tr_pred += [id2label[j] for j in ids]
        print(f"  transformer: {min(i + 32, len(texts))}/{len(texts)}", end="\r")
print()

pd.DataFrame({"clean_tweet": texts, "gold": y, "baseline": base_pred,
              "transformer_seed42": tr_pred}).to_csv(
    f"{OUT}/test_predictions_seed42.csv", index=False)

cm_b = confusion_matrix(y, base_pred, labels=LABELS)
cm_t = confusion_matrix(y, tr_pred, labels=LABELS)
b_ok = np.array(base_pred) == np.array(y)
t_ok = np.array(tr_pred) == np.array(y)
check = {
    "baseline": {"macro_f1": f1_score(y, base_pred, average="macro"),
                 "accuracy": accuracy_score(y, base_pred),
                 "per_class_f1": dict(zip(LABELS, f1_score(y, base_pred, average=None, labels=LABELS).tolist())),
                 "confusion_counts": cm_b.tolist()},
    "transformer_seed42": {"macro_f1": f1_score(y, tr_pred, average="macro"),
                           "accuracy": accuracy_score(y, tr_pred),
                           "per_class_f1": dict(zip(LABELS, f1_score(y, tr_pred, average=None, labels=LABELS).tolist())),
                           "confusion_counts": cm_t.tolist()},
    "outcome_groups": {"both_correct": int((b_ok & t_ok).sum()),
                       "transformer_only": int((~b_ok & t_ok).sum()),
                       "baseline_only": int((b_ok & ~t_ok).sum()),
                       "both_wrong": int((~b_ok & ~t_ok).sum())},
    "rows_order": LABELS,
}
json.dump(check, open(f"{OUT}/paper_figures_check.json", "w"), indent=2)
print(json.dumps({k: (v["macro_f1"] if "macro_f1" in v else v) for k, v in check.items() if k != "rows_order"}, indent=2))

# ── Figure 2: confusion matrices, shared colour scale ──────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
for ax, cm, title in [(axes[0], cm_b, "TF-IDF + Logistic Regression"),
                      (axes[1], cm_t, "Afro-XLM-R base (seed 42)")]:
    rn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{rn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=9, color="white" if rn[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), NAMES)
    ax.set_yticks(range(3), NAMES if ax is axes[0] else [])
    ax.set_xlabel("Predicted")
    ax.set_title(title, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("Gold")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Share of gold class")
fig.savefig(f"{OUT}/paper_fig2_confusion.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: per-class F1, baseline vs 3-seed transformer mean ─
# Transformer values are the 3-seed means and population std from Table 5.3.
base_f1 = [check["baseline"]["per_class_f1"][l] for l in LABELS] + [check["baseline"]["macro_f1"]]
tr_mean = [0.7056, 0.3333, 0.6240, 0.5543]
tr_std = [0.0222, 0.0081, 0.0107, 0.0068]
cats = NAMES + ["Macro F1"]
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(cats))
w = 0.36
b1 = ax.bar(x - w / 2, base_f1, w, label="TF-IDF + LR", color="#B8B8C8")
b2 = ax.bar(x + w / 2, tr_mean, w, yerr=tr_std, capsize=3, label="Afro-XLM-R base (mean of 3 seeds)", color="#3D6FB6")
for bars, vals, errs in [(b1, base_f1, [0] * 4), (b2, tr_mean, tr_std)]:
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.015, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x, cats)
ax.set_ylabel("F1 (test)")
ax.set_ylim(0, 0.85)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/paper_fig3_f1.png", dpi=300)
plt.close(fig)
print("Saved figures to reports/")
