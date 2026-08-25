# Fitbit: Calorie Burn Prediction & Workout Pattern Clustering

## Project Overview

This project uses Machine Learning to predict calories burned during workout sessions and identify hidden workout behavior patterns using Fitbit fitness data.

The project combines:

- Supervised Learning
- Unsupervised Learning
- PCA Dimensionality Reduction
- KMeans Clustering
- Streamlit Deployment

## Domain

Fitness Analytics / Health Tech / Machine Learning

## Objectives

1. Predict calories burned per workout session.
2. Compare multiple regression algorithms.
3. Identify hidden workout behavior patterns.
4. Evaluate clustering quality using Silhouette Score.
5. Deploy the prediction model using Streamlit.

## Dataset Features

The dataset contains:

- Age
- Gender
- Weight
- Height
- BMI
- Fat Percentage
- Maximum BPM
- Average BPM
- Resting BPM
- Session Duration
- Workout Type
- Water Intake
- Workout Frequency
- Experience Level
- Calories Burned

## Machine Learning Models

The following regression models were compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- Decision Tree
- Random Forest
- SVR
- XGBoost

## Regression Evaluation

The models were evaluated using:

- MAE
- RMSE
- R² Score

### Best Model

XGBoost achieved the best performance in the current experiment:

- MAE: 3.81
- RMSE: 5.85
- R²: 0.9989

## Clustering

Workout behavior patterns were identified using:

- Standard Scaling
- PCA
- KMeans
- Silhouette Score

Hierarchical Clustering and DBSCAN were also implemented for comparison.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard with:

### Calorie Prediction
Users can enter workout information and receive a predicted calorie burn value.

### Model Performance
Displays regression model comparison using MAE, RMSE and R².

### Workout Clusters
Displays cluster profiles, PCA visualization and Silhouette analysis.

## Project Structure

fitbit-calorie-burn-prediction/

├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── visuals/
│
└── src/

## How to Run

### 1. Clone the repository

git clone <YOUR_GITHUB_URL>

### 2. Create virtual environment

python3 -m venv .venv

### 3. Activate environment

source .venv/bin/activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Run regression

python main.py --task regression --data data/raw/Fitbit_dataset.csv

### 6. Run clustering

python main.py --task clustering --data data/raw/Fitbit_dataset.csv

### 7. Run Streamlit

streamlit run app.py

## Business Use Cases

- Real-time calorie estimation
- Personalized fitness coaching
- Workout recommendations
- User segmentation
- Nutrition planning
- Fitness device optimization

## Technologies

Python  
Pandas  
NumPy  
Scikit-learn  
XGBoost  
Matplotlib  
Seaborn  
Streamlit  
Git/GitHub