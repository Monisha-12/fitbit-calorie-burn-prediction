# fitbit-calorie-burn-prediction

# Fitbit: Calorie Burn Prediction & Workout Pattern Clustering

## 📌 Project Overview
This project focuses on building a machine learning system using Fitbit-like fitness data to:

- Predict calories burned during workout sessions (Regression)
- Identify hidden workout behavior patterns (Clustering)

The solution combines supervised and unsupervised learning techniques to generate actionable insights for fitness applications.

---

## 🎯 Problem Statement
Accurate calorie estimation is essential in modern fitness applications. While wearable devices capture physiological signals, additional contextual factors influence calorie burn.

This project aims to:
- Predict calorie expenditure using regression models
- Identify user workout patterns without labeled data using clustering techniques

---

## 🧠 Approach

### 🔹 Task 1: Regression (Calorie Prediction)
- Target: `Calories_Burned (kcal)`
- Models Used:
  - Linear Regression
  - Ridge / Lasso
  - KNN
  - Decision Tree
  - Random Forest
  - SVR
  - XGBoost

### 🔹 Task 2: Clustering (Workout Segmentation)
- Removed:
  - `Workout_Type` (label)
  - `Calories_Burned (kcal)` (target)
- Techniques:
  - Feature Scaling
  - PCA (Dimensionality Reduction)
  - KMeans Clustering
- Evaluation:
  - Silhouette Score

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost

---

## 📊 Key Results

### 🔹 Regression Performance
| Model | R² Score |
|------|---------|
| Random Forest | ~0.90+ |
| XGBoost | ~0.90+ |
| Linear Models | ~0.92 |

> Note: Initial near-perfect scores (~0.99) were due to **data leakage**, which was resolved by removing derived features.

---

### 🔹 Clustering Results
- Algorithm: KMeans (k=3)
- Silhouette Score: ≥ 0.15 (acceptable for real-world data)

#### Identified Clusters:
- **Low Intensity Users** → Low BPM, short duration
- **Moderate Users** → Balanced workouts
- **High Intensity Users** → High BPM, long sessions

---

## ⚠️ Data Leakage Handling
Features such as:
- `Base_MET`
- `Effective_MET`
- `HR_Intensity`

were identified as derived features closely related to calorie computation.

These were removed to ensure realistic model performance and generalization.

---

## 📁 Project Structure
fitbit-calorie-burn-prediction/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_eda.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_regression.py
│ ├── train_clustering.py
│
├── reports/
│ ├── regression_model_results.csv
│ ├── cluster_feature_means.csv
│
├── visuals/
│ ├── pca_clusters.png
│
├── main.py
├── requirements.txt
└── README.md


---

## 📈 Key Insights
- Heart rate and session duration are the strongest predictors of calorie burn
- Users naturally cluster into intensity groups without labeled workout types
- Real-world fitness data has overlapping patterns, making clustering challenging but meaningful

---

## 🎯 Business Use Cases
- Real-time calorie prediction in fitness apps
- Personalized workout recommendations
- User segmentation for engagement strategies
- Product insights for wearable device companies

---

## 🚀 How to Run

```bash
git clone https://github.com/Monisha-12/fitbit-calorie-burn-prediction
cd fitbit-calorie-burn-prediction

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python main.py