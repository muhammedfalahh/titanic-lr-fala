# Titanic ML - Logistic Regression Baseline & Iterations

Instead of throwing massive neural nets or complex algorithms at an 890-row tabular dataset, this repository provides a strict validation pipeline using a linear model (Logistic Regression) to track how deliberate feature engineering improves accuracy.

---

## The Results

Evaluated using 5-fold Stratified Cross-Validation on the training set to prevent data leakage:

| Iteration | Model | Key Changes | 5-Fold CV Accuracy | Test Survival Ratio |
|---|---|---|---|---|
| **v1 (Baseline)** | Logistic Regression | Raw features (`Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch`, `Embarked`), global median imputation | **79.35%** | 37.3% (156 / 418) |
| **v2 (Feature Eng.)** | Logistic Regression | Extracted `Title`, engineered `FamilySize`, smart title-based `Age` imputation | **83.05%** (+3.7%) | 36.8% (154 / 418) |

> **Key takeaway:** We gained **+3.7% accuracy** without touching a single hyperparameter or altering the underlying algorithm—purely by giving the model cleaner mathematical representations of the historical disaster.

---

## What Actually Moved the Needle

1. **Title Extraction from `Name`**:
   The historical rule of "women and children first" misled raw linear models because young boys are categorized as `male` (which normally carries a strong negative survival coefficient). Extracting the title `Master` separated boys from adult men (`Mr`), allowing the model to correctly identify child survival signals.

2. **Title-Based Imputation**:
   Instead of imputing missing ages with a global median (~28 years old), we calculated the median age per `Title` (e.g., ~3.5 years old for `Master`) using only the training split to avoid leakage.

3. **`FamilySize` Metric**:
   Derived as `SibSp + Parch + 1`. Solo travelers had lower survival rates, small families (2–4 members) had the highest survival rates, and large families suffered due to evacuation logistics on crowded decks.

---

## Project Structure

```text
titanic-lr/
├── train.csv                # Kaggle training dataset (contains labels)
├── test.csv                 # Kaggle test dataset (labels withheld)
├── main.py                  # End-to-end data processing, CV, training, and inference
├── submission_lr1.csv       # Baseline predictions (79.35% CV)
├── submission_lr2.csv       # Feature-engineered predictions (83.05% CV)
└── README.md                # Project documentation and run guide
```

---

## Quickstart / How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/muhammedfalahh/titanic-lr-fala.git](https://github.com/muhammedfalahh/titanic-lr-fala.git)
cd titanic-lr-fala
```

### 2. Set up a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*Note: If PowerShell restricts scripts, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.*

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required libraries
```bash
pip install pandas scikit-learn numpy
```

### 4. Ensure Kaggle CSV files are in place
Download `train.csv` and `test.csv` from Kaggle and verify they sit directly inside the root directory beside `main.py`.

### 5. Execute the pipeline
```bash
python main.py
```

### Expected Terminal Output
```plaintext
Average CV Accuracy: 83.05%
Predicted Survived: 154 | Predicted Deceased: 264
First 5 Predictions:
   PassengerId  Survived
0          892         0
1          893         1
2          894         0
3          895         0
4          896         1
```

The script automatically generates `submission_lr2.csv` in the root folder, ready for immediate upload to the Kaggle competition page.

---