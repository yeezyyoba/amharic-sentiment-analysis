# Amharic Sentiment Analysis with Afro-XLM-R

## Final Project Report

**Author**: Eyob Nebiyou
**Institution**: Addis Ababa University, CNCS
**Date**: September 2026
**GitHub**: https://github.com/yeezyyoba/amharic-sentiment-analysis

---

## 1. Executive Summary

This project investigates sentiment classification for Amharic social media text using the AfriSenti Amharic dataset. The study establishes a TF-IDF + Logistic Regression baseline and compares it with a fine-tuned Afro-XLM-R transformer model.

The baseline achieves a test Macro F1 score of 0.4640. The fine-tuned Afro-XLM-R model achieves a mean Macro F1 of 0.5543 ± 0.0068 across three random seeds (42, 123, and 456), representing an improvement of 9.03 percentage points over the baseline.

The project also includes exploratory data analysis, Amharic-specific preprocessing, tokenization analysis, error analysis, per-class evaluation, and qualitative inspection of transformer attention patterns.

---

## 2. Problem Statement

Amharic is one of the major languages of Ethiopia and has a comparatively limited amount of publicly available NLP resources. Sentiment analysis for Amharic social media text can support applications such as public opinion analysis, customer feedback analysis, and social media monitoring.

The objective of this project is to evaluate how effectively a multilingual transformer model can classify Amharic tweets into three sentiment categories:

- Negative
- Neutral
- Positive

The study compares transformer-based classification against a traditional TF-IDF-based machine learning baseline.

---

## 3. Dataset

The project uses the Amharic subset of the AfriSenti dataset.

| Property | Detail |
|---|---|
| Dataset | AfriSenti |
| Language | Amharic (`amh`) |
| Training samples | 5,984 |
| Validation samples | 1,497 |
| Test samples | 1,999 |
| Classes | Negative, Neutral, Positive |

The training split contains approximately 51.9% neutral, 25.9% negative, and 22.3% positive examples.

The test split contains 1,337 negative, 438 positive, and 224 neutral examples.

---

## 4. Methodology

### 4.1 Exploratory Data Analysis

The dataset was examined to understand:

- Split sizes and class distributions
- Common tokens and vocabulary patterns
- Mixed-script text
- URLs, mentions, and hashtags
- Potentially ambiguous examples

Manual inspection of selected examples suggests that some neutral examples may have ambiguous or debatable sentiment labels.

### 4.2 Preprocessing

The preprocessing pipeline includes:

- Removal of URLs, mentions, and hashtags
- Normalization of selected Ethiopic characters
- Lowercasing of Latin characters
- Removal of duplicate cleaned tweets within each split
- Removal of invalid or missing entries where applicable

The resulting processed splits contain 5,968 training samples, 1,496 validation samples, and 1,999 test samples.

### 4.3 TF-IDF Baseline

A traditional TF-IDF representation was used with several classical classifiers. The vectorizer was fitted only on the training data and then applied to the validation and test sets.

Logistic Regression achieved the strongest baseline validation performance and was used as the primary baseline.

Its test Macro F1 score was:

**0.4640**

Macro F1 was selected as the primary evaluation metric because it gives equal importance to each sentiment class despite class imbalance.

### 4.4 Transformer Fine-Tuning

The transformer approach uses Afro-XLM-R, a multilingual model designed for African languages.

The model was fine-tuned for three-class Amharic sentiment classification. Model selection was performed using the validation set, with Macro F1 as the primary metric.

The final multi-seed evaluation used seeds:

- 42
- 123
- 456

The recorded test Macro F1 scores were:

| Seed | Macro F1 |
|---|---:|
| 42 | 0.5620 |
| 123 | 0.5553 |
| 456 | 0.5455 |
| **Mean** | **0.5543** |
| **Std.** | **0.0068** |

---

## 5. Results

The transformer improves over the TF-IDF baseline on the test set.

| Model | Test Macro F1 |
|---|---:|
| TF-IDF + Logistic Regression | 0.4640 |
| Afro-XLM-R | 0.5543 ± 0.0068 |

The mean improvement over the baseline is **9.03 percentage points**.

The seed-42 transformer evaluation also provides a detailed per-class analysis and confusion matrix.

---

## 6. Error Analysis

Error analysis was conducted to examine cases where the baseline and transformer models succeed or fail.

On the test set:

- Both models were correct on 750 examples.
- The transformer was correct while the baseline was incorrect on 474 examples.
- The baseline was correct while the transformer was incorrect on 249 examples.
- Both models were incorrect on 526 examples.

The analysis indicates that sentiment classification remains challenging for examples containing ambiguous language, contextual sentiment, and cases where sentiment cannot be identified from isolated lexical cues.

The transformer also improves on several classes where the TF-IDF baseline makes substantial errors, although errors remain across all three sentiment categories.

---

## 7. Attention Visualization

Selected examples were inspected using token-level attention patterns from the final transformer layer, averaged across attention heads.

In the selected negative examples, higher attention was observed around words such as ወጠጤ and ሽፍታ. In selected positive examples, attention was distributed across words such as ምህረት and በረከት.

These visualizations are illustrative of attention patterns in the selected examples and should not be interpreted as definitive explanations of the model's predictions.

---

## 8. Limitations

Several limitations should be considered:

- The dataset is relatively small compared with datasets available for high-resource languages.
- The dataset contains class imbalance, particularly in the test split.
- Some examples may contain ambiguous or debatable sentiment labels.
- Evaluation is based on a single publicly available dataset and may not represent all forms of contemporary Amharic social media text.
- Attention visualizations are based on selected examples and should not be treated as definitive model explanations.
- The reported multi-seed results reflect three seeds and may vary with different training configurations or hardware environments.

---

## 9. Conclusion

This project evaluates Amharic sentiment classification using both a traditional TF-IDF baseline and a fine-tuned Afro-XLM-R transformer.

The TF-IDF + Logistic Regression baseline achieves a test Macro F1 of 0.4640, while Afro-XLM-R achieves a mean Macro F1 of 0.5543 ± 0.0068 across three seeds.

The results demonstrate the potential of multilingual transformer models for Amharic sentiment analysis while also highlighting the challenges associated with limited data, class imbalance, and ambiguous social media language.

The repository provides the complete experimental workflow, including data exploration, preprocessing, baseline modeling, transformer fine-tuning, evaluation, and error analysis.

---

## 10. References

- AfriSenti dataset: Amharic sentiment analysis benchmark.
- Afro-XLM-R: Multilingual transformer model for African languages.
- TF-IDF: Term Frequency–Inverse Document Frequency representation.
- Logistic Regression: Classical supervised classification baseline.
