# 📉 ChurnSight — Customer Churn Prediction ML Dashboard

A complete end-to-end Machine Learning pipeline for Customer Churn Prediction,
built with Streamlit, Scikit-learn, XGBoost, and Plotly.

## 🚀 How to Run Locally

```bash
# 1. Clone / download the project folder
cd churn_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at http://localhost:8501

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. Push your project to a **GitHub repository**
2. Go to https://share.streamlit.io
3. Click **"New app"**
4. Select your repo, branch (`main`), and file (`app.py`)
5. Click **Deploy** — done! 🎉

Your app gets a public URL: `https://your-app.streamlit.app`

---

## 📁 Project Structure

```
churn_app/
├── app.py              ← Main Streamlit application
├── generate_data.py    ← Synthetic dataset generator
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🔬 ML Pipeline (20 Steps Implemented)

| Step | Task |
|------|------|
| 1-3  | Libraries, dataset load, shape/dtype/stats check |
| 4    | Missing value imputation (median) |
| 5    | Duplicate removal |
| 6    | Univariate analysis (histograms + count plots) |
| 7    | Bivariate analysis (churn vs features) |
| 8    | Correlation heatmap |
| 9    | Outlier detection & IQR clipping |
| 10   | Label Encoding + One-Hot Encoding |
| 11   | StandardScaler / MinMaxScaler |
| 12   | X / y separation |
| 13   | Train-test split (stratified) |
| 14   | SMOTE for class imbalance |
| 15   | 7 models: LR, KNN, DT, RF, SVC, GB, XGBoost |
| 16   | Acc, Precision, Recall, F1, ROC-AUC, CM |
| 17   | GridSearchCV (RF) + RandomizedSearchCV (XGB) |
| 18   | Model comparison table + charts |
| 19   | Best model selection |
| 20   | Key insights + business recommendations |

---

## 🎯 App Pages

- **Overview & EDA** — Dataset exploration, distributions, bivariate analysis, correlation
- **Preprocessing** — Missing values, duplicates, outliers, encoding, SMOTE
- **Model Training** — 7 algorithms trained, feature importance
- **Model Evaluation** — Metrics, ROC curves, confusion matrices, cross-validation
- **Hyperparameter Tuning** — GridSearchCV & RandomizedSearchCV results
- **Results & Insights** — Model comparison, key churn factors, business recommendations
- **Predict Churn** — Live prediction with risk gauge for any customer profile

---

## 📦 Tech Stack

- **Streamlit** — Web app framework
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML models, preprocessing, evaluation
- **XGBoost** — Gradient boosting classifier
- **Imbalanced-learn** — SMOTE oversampling
- **Plotly** — Interactive visualizations
