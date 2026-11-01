# Titanic ML -Logistic Regression Baseline & Iterations

Instead of trhwoing massive nueral nets or complex algorithms at an 890-row tabular dataset, This is a strict validation pipeline using linear model (Logistic Regression) and tracked how feature engineering moves the accuracy.

-----

## The Results

Ran 5-fold Stratified Cross-Validation on the training set to prevent data leakage and evaluate real performance:

| Iteration | Model | Key Changes | 5-Fold CV Accuracy | Test Survival Ratio |
|---|---|---|---|---|
| **v1 (Baseline)** | Logistic Regression | Raw features (`Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch`, `Embarked`), global median imputation | **79.35%** | 37.3% (156 / 418) |
| **v2 (Feature Eng.)** | Logistic Regression | Extracted `Title`, engineered `FamilySize`, smart title-based `Age` imputation | **83.05%** (+3.7%) | 36.8% (154 / 418) |

> **Key takeaway:** We gained **+3.7% accuracy** without touching a single hyperparameter or changing the underlying algorithm. We just gave the model better mathematical representations of the historical event.

---

## What Actually Moved the Needle

1. **Title Extraction from `Name`**:
   The historical rule of "women and children first" broke raw linear models because young boys are labeled `male` (which usually pushes survival odds down). Extracting `Master` separated young boys from adult men (`Mr`), allowing the model to accurately predict their survival.
2. **Title-Based Imputation**:
   Instead of filling missing ages with a global median (~28 years old), we filled missing ages using the median age of that passenger's specific `Title` (e.g., ~3.5 years old for a missing `Master`).
3. **`FamilySize` Metric**:
   `SibSp + Parch + 1`. Solo travelers had lower survival rates, small families (2–4) had the highest survival rates, and large families suffered due to chaos on the decks.

---

## Project Structure

```text
titanic-lr/
├── train.csv                # Kaggle training data
├── test.csv                 # Kaggle test data
├── main.py                  # Full data cleaning, training, and inference pipeline
├── submission_lr1.csv       # Dumb Baseline
├── submission_lr2.csv       # Final predictions.
└── README.md                # Project documentation