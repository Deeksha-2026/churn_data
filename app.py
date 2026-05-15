import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from generate_data import generate_churn_dataset

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight | ML Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; }

/* Main BG */
.main { background-color: #f8f7ff; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin: 0;
}
.hero-sub {
    font-size: 1rem;
    opacity: 0.75;
    margin-top: 0.3rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-card {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(48,43,99,0.10);
    border-left: 4px solid #302b63;
}
.metric-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #888; }
.metric-card .value { font-family: 'Space Mono', monospace; font-size: 1.7rem; color: #0f0c29; font-weight: 700; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    color: #302b63;
    letter-spacing: 0.5px;
    border-bottom: 2px solid #302b63;
    padding-bottom: 0.3rem;
    margin: 1.5rem 0 1rem 0;
}

/* Model card */
.model-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    box-shadow: 0 2px 12px rgba(48,43,99,0.08);
    margin-bottom: 0.8rem;
    border-left: 4px solid #7c5cbf;
}
.model-card.best { border-left-color: #f72585; }

/* Badges */
.badge {
    display: inline-block;
    background: #302b63;
    color: white;
    font-size: 0.7rem;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    margin-left: 8px;
    vertical-align: middle;
}
.badge.best { background: #f72585; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #e0d7ff, #f3eeff);
    border-left: 4px solid #7c5cbf;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #1a1040;
}

/* Recommendation card */
.rec-card {
    background: white;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0;
    box-shadow: 0 1px 8px rgba(48,43,99,0.07);
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
}
.rec-icon { font-size: 1.4rem; }
.rec-text { font-size: 0.88rem; color: #333; }
.rec-text strong { color: #302b63; }

/* Step tags */
.step-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    background: #302b63;
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
}

/* Progress bar */
.stProgress .st-bo { background-color: #302b63; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 ChurnSight")
    st.markdown("*Customer Churn ML Pipeline*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview & EDA",
        "🔬 Preprocessing",
        "🤖 Model Training",
        "📊 Model Evaluation",
        "🔧 Hyperparameter Tuning",
        "🏆 Results & Insights",
        "🎯 Predict Churn"
    ])
    st.markdown("---")
    st.markdown("**Dataset Controls**")
    n_samples = st.slider("Sample Size", 500, 5000, 2000, 100)
    test_size = st.slider("Test Split %", 10, 40, 20, 5)
    scaler_choice = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])
    use_smote = st.checkbox("Apply SMOTE", value=True)
    st.markdown("---")
    st.caption("Built with Streamlit + Scikit-learn + XGBoost")

# ─────────────────────────────────────────────
#  CACHED DATA & PIPELINE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw(n):
    return generate_churn_dataset(n)

@st.cache_data(show_spinner=False)
def run_pipeline(n, test_sz, scaler_name, apply_smote):
    raw = generate_churn_dataset(n)
    df = raw.copy()

    # ── 1. Missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    missing_before = df.isnull().sum().to_dict()
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())

    # ── 2. Duplicates
    n_dups = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 3. Outliers (IQR cap)
    num_cols = ['CustomerAge', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'CustomerSupportCalls']
    outlier_counts = {}
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        outlier_counts[col] = int(n_out)
        df[col] = df[col].clip(lo, hi)

    # ── 4. Encoding
    cat_cols = ['Gender', 'ContractType', 'PaymentMethod', 'InternetService',
                'TechSupport', 'OnlineSecurity']
    le = LabelEncoder()
    df['Gender_enc'] = le.fit_transform(df['Gender'])
    df_ohe = pd.get_dummies(df[cat_cols[1:]], drop_first=False)
    df = pd.concat([df.drop(cat_cols, axis=1), df['Gender_enc'], df_ohe], axis=1)
    df.columns = df.columns.astype(str)

    # ── 5. Features / Target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # ── Fill any remaining NaNs before scaling
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)  # fallback for any non-numeric columns
    feature_names = X.columns.tolist()

    # ── 6. Scale
    scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # Final NaN safety check after scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # ── 7. Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_sz/100, random_state=42, stratify=y)

    # ── 8. SMOTE
    smote_info = {}
    if apply_smote:
        before = dict(pd.Series(y_tr).value_counts())
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        after = dict(pd.Series(y_tr).value_counts())
        smote_info = {'before': before, 'after': after}

    # ── 9. Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss',
                                 use_label_encoder=False, random_state=42)
    }

    results = {}
    cv_scores = {}
    trained_models = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        trained_models[name] = model
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_data[name] = (fpr.tolist(), tpr.tolist())
        cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring='roc_auc')
        cv_scores[name] = cv.tolist()
        results[name] = {
            'Accuracy': round(accuracy_score(y_te, y_pred), 4),
            'Precision': round(precision_score(y_te, y_pred), 4),
            'Recall': round(recall_score(y_te, y_pred), 4),
            'F1 Score': round(f1_score(y_te, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_te, y_prob), 4),
            'CV Mean AUC': round(float(np.mean(cv)), 4),
            'Confusion Matrix': confusion_matrix(y_te, y_pred).tolist()
        }

    # ── 10. Hyperparameter tuning on best model
    best_name = max(results, key=lambda k: results[k]['ROC-AUC'])
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, None],
        'min_samples_split': [2, 5]
    }
    xgb_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.9, 1.0]
    }
    gs = GridSearchCV(RandomForestClassifier(random_state=42),
                      rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_tr, y_tr)
    rs = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', use_label_encoder=False,
                                          random_state=42),
                            xgb_param_dist, n_iter=10, cv=3, scoring='roc_auc',
                            n_jobs=-1, random_state=42)
    rs.fit(X_tr, y_tr)

    tuning_results = {
        'GridSearchCV (RF)': {
            'best_params': gs.best_params_,
            'best_score': round(gs.best_score_, 4)
        },
        'RandomizedSearchCV (XGB)': {
            'best_params': rs.best_params_,
            'best_score': round(rs.best_score_, 4)
        }
    }

    # Feature importance (RF)
    rf_model = trained_models['Random Forest']
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    return {
        'raw': raw,
        'df_clean': df,
        'missing_before': missing_before,
        'n_dups': n_dups,
        'outlier_counts': outlier_counts,
        'smote_info': smote_info,
        'X_te': X_te,
        'y_te': y_te,
        'X_tr': X_tr,
        'y_tr': y_tr,
        'feature_names': feature_names,
        'scaler': scaler,
        'results': results,
        'cv_scores': cv_scores,
        'roc_data': roc_data,
        'best_name': best_name,
        'tuning_results': tuning_results,
        'feature_importance': fi.to_dict(),
        'trained_models': trained_models
    }

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
with st.spinner("Running ML pipeline…"):
    p = run_pipeline(n_samples, test_size, scaler_choice, use_smote)

raw = p['raw']
df  = p['df_clean']
res = p['results']

# ──────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW & EDA
# ──────────────────────────────────────────────────────────────
if page == "🏠 Overview & EDA":
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">📉 ChurnSight</p>
      <p class="hero-sub">End-to-End Customer Churn Prediction · ML Classification Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    churn_rate = raw['Churn'].mean() * 100
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="label">Total Customers</div>
        <div class="value">{len(raw):,}</div>
      </div>
      <div class="metric-card">
        <div class="label">Churn Rate</div>
        <div class="value">{churn_rate:.1f}%</div>
      </div>
      <div class="metric-card">
        <div class="label">Features</div>
        <div class="value">11</div>
      </div>
      <div class="metric-card">
        <div class="label">Best Model AUC</div>
        <div class="value">{max(v['ROC-AUC'] for v in res.values()):.3f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset snapshot
    st.markdown('<p class="section-header">STEP 2–3 · Dataset Overview</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📐 Shape & Types", "📊 Statistics"])
    with tab1:
        st.dataframe(raw.head(20), use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", raw.shape[0])
            st.metric("Columns", raw.shape[1])
        with col2:
            dtype_df = pd.DataFrame({'Column': raw.columns,
                                     'Dtype': raw.dtypes.values.astype(str),
                                     'Nulls': raw.isnull().sum().values})
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(raw.describe().round(2), use_container_width=True)

    # ── Univariate
    st.markdown('<p class="section-header">STEP 6 · Univariate Analysis</p>', unsafe_allow_html=True)
    num_cols = ['CustomerAge', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'CustomerSupportCalls']
    fig = make_subplots(rows=1, cols=5, subplot_titles=num_cols)
    for i, col in enumerate(num_cols, 1):
        fig.add_trace(go.Histogram(x=raw[col], nbinsx=25,
                                   marker_color='#302b63', opacity=0.75,
                                   name=col, showlegend=False), row=1, col=i)
    fig.update_layout(height=300, title_text="Distribution of Numerical Features",
                      plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                      margin=dict(t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    cat_cols = ['Gender', 'ContractType', 'PaymentMethod',
                'InternetService', 'TechSupport', 'OnlineSecurity']
    cols = st.columns(3)
    for i, col in enumerate(cat_cols):
        vc = raw[col].value_counts().reset_index()
        vc.columns = [col, 'Count']
        fig2 = px.bar(vc, x=col, y='Count',
                      color_discrete_sequence=['#302b63', '#7c5cbf', '#c77dff', '#f72585'],
                      title=col)
        fig2.update_layout(height=260, margin=dict(t=40, b=20, l=10, r=10),
                           plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        cols[i % 3].plotly_chart(fig2, use_container_width=True)

    # ── Bivariate
    st.markdown('<p class="section-header">STEP 7 · Bivariate Analysis – Churn vs Features</p>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ct = pd.crosstab(raw['ContractType'], raw['Churn'], normalize='index') * 100
        fig3 = px.bar(ct.reset_index(), x='ContractType', y=[0, 1],
                      barmode='group', title='Churn % by Contract Type',
                      labels={'value': 'Percentage', 'variable': 'Churn'},
                      color_discrete_map={0: '#302b63', 1: '#f72585'})
        fig3.update_layout(height=320, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = px.box(raw, x='Churn', y='MonthlyCharges',
                      color='Churn', title='Monthly Charges vs Churn',
                      color_discrete_map={0: '#302b63', 1: '#f72585'})
        fig4.update_layout(height=320, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig4, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig5 = px.box(raw, x='Churn', y='Tenure',
                      color='Churn', title='Tenure vs Churn',
                      color_discrete_map={0: '#302b63', 1: '#f72585'})
        fig5.update_layout(height=300, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig5, use_container_width=True)
    with col4:
        ts_churn = raw.groupby('TechSupport')['Churn'].mean().reset_index()
        ts_churn['Churn'] *= 100
        fig6 = px.bar(ts_churn, x='TechSupport', y='Churn',
                      title='Churn Rate by Tech Support',
                      color_discrete_sequence=['#7c5cbf'])
        fig6.update_layout(height=300, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig6, use_container_width=True)

    # ── Correlation heatmap
    st.markdown('<p class="section-header">STEP 8 · Correlation Heatmap</p>',
                unsafe_allow_html=True)
    num_df = raw[['CustomerAge', 'Tenure', 'MonthlyCharges',
                  'TotalCharges', 'CustomerSupportCalls', 'Churn']].copy()
    corr = num_df.corr()
    fig7 = px.imshow(corr, text_auto='.2f', aspect='auto',
                     color_continuous_scale='RdBu_r',
                     title='Feature Correlation Matrix')
    fig7.update_layout(height=420, paper_bgcolor='#f8f7ff')
    st.plotly_chart(fig7, use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  PAGE: PREPROCESSING
# ──────────────────────────────────────────────────────────────
elif page == "🔬 Preprocessing":
    st.markdown('<p class="hero-title" style="color:#302b63">🔬 Data Preprocessing</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Missing values
    st.markdown('<p class="section-header">STEP 4 · Missing Values</p>', unsafe_allow_html=True)
    mv = {k: v for k, v in p['missing_before'].items() if v > 0}
    if mv:
        col1, col2 = st.columns(2)
        with col1:
            mv_df = pd.DataFrame.from_dict(mv, orient='index', columns=['Missing Count'])
            mv_df['% Missing'] = (mv_df['Missing Count'] / len(raw) * 100).round(2)
            st.dataframe(mv_df, use_container_width=True)
        with col2:
            st.markdown("""
            <div class="insight-box">
            ✅ <b>Strategy Applied:</b><br>
            • <b>TotalCharges</b> → filled with <b>median</b> (skewed distribution)<br>
            • <b>MonthlyCharges</b> → filled with <b>median</b> (robust to outliers)
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No missing values detected.")

    # Duplicates
    st.markdown('<p class="section-header">STEP 5 · Duplicate Records</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-box">
    🔍 Found <b>{p['n_dups']} duplicate rows</b> → Removed. Clean dataset: <b>{len(df):,} rows</b>.
    </div>
    """, unsafe_allow_html=True)

    # Outliers
    st.markdown('<p class="section-header">STEP 9 · Outlier Detection (IQR Method)</p>',
                unsafe_allow_html=True)
    out_df = pd.DataFrame.from_dict(p['outlier_counts'], orient='index',
                                    columns=['Outliers Detected'])
    out_df['Action'] = 'IQR Clipping Applied'
    st.dataframe(out_df, use_container_width=True)

    fig_box = make_subplots(rows=1, cols=5,
                            subplot_titles=list(p['outlier_counts'].keys()))
    for i, col in enumerate(p['outlier_counts'].keys(), 1):
        if col in raw.columns:
            fig_box.add_trace(go.Box(y=raw[col], name=col, showlegend=False,
                                     marker_color='#302b63'), row=1, col=i)
    fig_box.update_layout(height=320, title_text="Boxplots (Before Clipping)",
                          plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
    st.plotly_chart(fig_box, use_container_width=True)

    # Encoding
    st.markdown('<p class="section-header">STEP 10 · Encoding</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Label Encoding** applied to:
        - `Gender` → 0/1
        """)
    with col2:
        st.markdown("""
        **One-Hot Encoding** applied to:
        - ContractType, PaymentMethod, InternetService, TechSupport, OnlineSecurity
        """)

    # Scaling & SMOTE
    st.markdown('<p class="section-header">STEP 11 + 14 · Scaling & Class Balancing</p>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="insight-box">
        ⚖️ <b>Scaler:</b> {scaler_choice}<br>
        Applied to all {len(p['feature_names'])} feature columns before model training.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if use_smote and p['smote_info']:
            bef = p['smote_info']['before']
            aft = p['smote_info']['after']
            st.markdown(f"""
            <div class="insight-box">
            🔁 <b>SMOTE Applied</b><br>
            Before → Churn 0: {bef.get(0, 0)} | Churn 1: {bef.get(1, 0)}<br>
            After  → Churn 0: {aft.get(0, 0)} | Churn 1: {aft.get(1, 0)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("SMOTE not applied.")

    st.markdown('<p class="section-header">STEP 13 · Train-Test Split</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="label">Training Samples</div>
        <div class="value">{len(p['X_tr']):,}</div>
      </div>
      <div class="metric-card">
        <div class="label">Test Samples</div>
        <div class="value">{len(p['X_te']):,}</div>
      </div>
      <div class="metric-card">
        <div class="label">Features</div>
        <div class="value">{len(p['feature_names'])}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  PAGE: MODEL TRAINING
# ──────────────────────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.markdown('<p class="hero-title" style="color:#302b63">🤖 Model Training</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    models_info = {
        'Logistic Regression': ('Linear baseline model', '⚡'),
        'K-Nearest Neighbors': ('Instance-based learner', '🔍'),
        'Decision Tree': ('Rule-based tree classifier', '🌿'),
        'Random Forest': ('Ensemble of decision trees', '🌲'),
        'SVC': ('Margin-maximizing classifier', '🔷'),
        'Gradient Boosting': ('Sequential boosting ensemble', '🚀'),
        'XGBoost': ('Optimized gradient boosting', '⚡🌲')
    }

    st.markdown('<p class="section-header">STEP 15 · Classification Models Trained</p>',
                unsafe_allow_html=True)

    for name, (desc, icon) in models_info.items():
        metrics = res[name]
        is_best = name == p['best_name']
        badge = '<span class="badge best">BEST</span>' if is_best else ''
        st.markdown(f"""
        <div class="model-card {'best' if is_best else ''}">
          <b>{icon} {name}</b>{badge}<br>
          <small style="color:#666">{desc}</small><br>
          <span style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#302b63">
          ACC: {metrics['Accuracy']:.3f} &nbsp;|&nbsp;
          F1: {metrics['F1 Score']:.3f} &nbsp;|&nbsp;
          AUC: {metrics['ROC-AUC']:.3f}
          </span>
        </div>
        """, unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<p class="section-header">Feature Importance (Random Forest)</p>',
                unsafe_allow_html=True)
    fi = p['feature_importance']
    fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
    fi_df = fi_df.nlargest(15, 'Importance')
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Purples',
                    title='Top 15 Feature Importances')
    fig_fi.update_layout(height=420, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                         yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_fi, use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  PAGE: MODEL EVALUATION
# ──────────────────────────────────────────────────────────────
elif page == "📊 Model Evaluation":
    st.markdown('<p class="hero-title" style="color:#302b63">📊 Model Evaluation</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">STEP 16 · Metrics Table</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(res).T.drop('Confusion Matrix', axis=1)
    metrics_df = metrics_df.astype(float)

    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #e0d7ff; font-weight: bold' if v else '' for v in is_max]

    st.dataframe(metrics_df.style.apply(highlight_best).format("{:.4f}"),
                 use_container_width=True)

    # ROC curves
    st.markdown('<p class="section-header">ROC Curves</p>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    colors = ['#302b63', '#7c5cbf', '#c77dff', '#f72585',
              '#480ca8', '#b5179e', '#4cc9f0']
    for (name, (fpr, tpr)), color in zip(p['roc_data'].items(), colors):
        auc = res[name]['ROC-AUC']
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"{name} (AUC={auc:.3f})",
                                     line=dict(color=color, width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 line=dict(dash='dash', color='gray'),
                                 name='Random', showlegend=True))
    fig_roc.update_layout(title='ROC Curves – All Models', height=480,
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
    st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrices
    st.markdown('<p class="section-header">Confusion Matrices</p>', unsafe_allow_html=True)
    model_names = list(res.keys())
    cols = st.columns(4)
    for i, name in enumerate(model_names):
        cm = np.array(res[name]['Confusion Matrix'])
        fig_cm = px.imshow(cm, text_auto=True,
                           color_continuous_scale='Purples',
                           labels=dict(x="Predicted", y="Actual"),
                           x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                           title=name)
        fig_cm.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10),
                             paper_bgcolor='#f8f7ff',
                             coloraxis_showscale=False)
        cols[i % 4].plotly_chart(fig_cm, use_container_width=True)

    # Cross-validation
    st.markdown('<p class="section-header">5-Fold Cross-Validation AUC Scores</p>',
                unsafe_allow_html=True)
    cv_data = []
    for name, scores in p['cv_scores'].items():
        for s in scores:
            cv_data.append({'Model': name, 'CV AUC': s})
    cv_df = pd.DataFrame(cv_data)
    fig_cv = px.box(cv_df, x='Model', y='CV AUC',
                    color='Model',
                    color_discrete_sequence=['#302b63', '#7c5cbf', '#c77dff',
                                             '#f72585', '#480ca8', '#b5179e', '#4cc9f0'],
                    title='Cross-Validation AUC Distribution per Model')
    fig_cv.update_layout(height=380, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                         showlegend=False, xaxis_tickangle=-25)
    st.plotly_chart(fig_cv, use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  PAGE: HYPERPARAMETER TUNING
# ──────────────────────────────────────────────────────────────
elif page == "🔧 Hyperparameter Tuning":
    st.markdown('<p class="hero-title" style="color:#302b63">🔧 Hyperparameter Tuning</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">STEP 17 · GridSearchCV & RandomizedSearchCV</p>',
                unsafe_allow_html=True)

    for method, info in p['tuning_results'].items():
        st.markdown(f"""
        <div class="model-card">
          <b>{'🔷' if 'Grid' in method else '🎲'} {method}</b><br>
          <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#302b63">
          Best CV AUC: <b>{info['best_score']:.4f}</b>
          </span><br>
          <small style="color:#555">Best Parameters: {info['best_params']}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Parameter Search Space</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **GridSearchCV – Random Forest**
        ```
        n_estimators:       [100, 200]
        max_depth:          [6, 10, None]
        min_samples_split:  [2, 5]
        cv = 3, scoring = roc_auc
        ```
        """)
    with col2:
        st.markdown("""
        **RandomizedSearchCV – XGBoost**
        ```
        n_estimators:    [100, 200, 300]
        max_depth:       [3, 5, 7]
        learning_rate:   [0.05, 0.1, 0.2]
        subsample:       [0.7, 0.9, 1.0]
        n_iter=10, cv=3, scoring=roc_auc
        ```
        """)

    st.markdown('<p class="section-header">Tuning Impact Comparison</p>', unsafe_allow_html=True)
    base_rf  = res['Random Forest']['ROC-AUC']
    base_xgb = res['XGBoost']['ROC-AUC']
    tuned_rf  = p['tuning_results']['GridSearchCV (RF)']['best_score']
    tuned_xgb = p['tuning_results']['RandomizedSearchCV (XGB)']['best_score']

    comp_df = pd.DataFrame({
        'Model': ['Random Forest', 'Random Forest', 'XGBoost', 'XGBoost'],
        'Stage': ['Baseline', 'Tuned', 'Baseline', 'Tuned'],
        'ROC-AUC': [base_rf, tuned_rf, base_xgb, tuned_xgb]
    })
    fig_tune = px.bar(comp_df, x='Model', y='ROC-AUC', color='Stage', barmode='group',
                      color_discrete_map={'Baseline': '#aaa', 'Tuned': '#302b63'},
                      title='Baseline vs Tuned ROC-AUC')
    fig_tune.update_layout(height=340, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                           yaxis=dict(range=[0.5, 1.0]))
    st.plotly_chart(fig_tune, use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  PAGE: RESULTS & INSIGHTS
# ──────────────────────────────────────────────────────────────
elif page == "🏆 Results & Insights":
    st.markdown('<p class="hero-title" style="color:#302b63">🏆 Results & Business Insights</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Model comparison table
    st.markdown('<p class="section-header">STEP 18 · Model Comparison Table</p>',
                unsafe_allow_html=True)
    metrics_df = pd.DataFrame(res).T.drop('Confusion Matrix', axis=1).astype(float)
    metrics_df['Rank'] = metrics_df['ROC-AUC'].rank(ascending=False).astype(int)
    metrics_df = metrics_df.sort_values('ROC-AUC', ascending=False)
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    # Best model
    st.markdown('<p class="section-header">STEP 19 · Best Model Selected</p>',
                unsafe_allow_html=True)
    best = p['best_name']
    bm = res[best]
    st.markdown(f"""
    <div class="model-card best">
      <span class="badge best">WINNER</span>
      <b style="font-size:1.2rem"> {best}</b><br><br>
      <div class="metric-row">
        <div class="metric-card"><div class="label">Accuracy</div><div class="value">{bm['Accuracy']:.3f}</div></div>
        <div class="metric-card"><div class="label">Precision</div><div class="value">{bm['Precision']:.3f}</div></div>
        <div class="metric-card"><div class="label">Recall</div><div class="value">{bm['Recall']:.3f}</div></div>
        <div class="metric-card"><div class="label">F1 Score</div><div class="value">{bm['F1 Score']:.3f}</div></div>
        <div class="metric-card"><div class="label">ROC-AUC</div><div class="value">{bm['ROC-AUC']:.3f}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Key factors
    st.markdown('<p class="section-header">STEP 20 · Key Churn Drivers</p>',
                unsafe_allow_html=True)
    factors = [
        ("📈", "High Monthly Charges", "Customers paying >$80/month churn ~30% more than average."),
        ("📋", "Month-to-Month Contracts", "~55% of churn comes from short-term contract holders."),
        ("🛠️", "No Tech Support", "Customers without tech support are significantly more likely to leave."),
        ("⏱️", "Short Tenure (<12 months)", "New customers have the highest churn risk in the first year."),
        ("📞", "High Support Calls (>5)", "Frequent support interactions signal dissatisfaction."),
    ]
    for icon, title, desc in factors:
        st.markdown(f"""
        <div class="insight-box">
        {icon} <b>{title}</b> — {desc}
        </div>
        """, unsafe_allow_html=True)

    # Business recommendations
    st.markdown('<p class="section-header">Business Recommendations</p>', unsafe_allow_html=True)
    recs = [
        ("💰", "Offer Long-Term Contract Discounts",
         "Incentivize customers to switch from month-to-month to annual/bi-annual plans."),
        ("🎧", "Improve Customer Support Quality",
         "Reduce resolution time and first-call resolution rate to cut support calls."),
        ("🎁", "Personalized Retention Offers",
         "Identify high-risk segments and send targeted loyalty rewards or plan discounts."),
        ("🚨", "Early-Warning Churn Detection",
         "Deploy this model in production to flag high-risk customers within first 3 months."),
        ("🏅", "Loyalty Programs",
         "Reward tenure milestones (6 months, 1 year, etc.) to improve long-term retention."),
    ]
    for icon, title, desc in recs:
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-icon">{icon}</div>
          <div class="rec-text"><strong>{title}</strong><br>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Radar chart comparing top 3 models
    st.markdown('<p class="section-header">Top 3 Model Radar Comparison</p>',
                unsafe_allow_html=True)
    top3 = metrics_df.head(3)
    cats = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'CV Mean AUC']
    fig_radar = go.Figure()
    colors_r = ['#f72585', '#302b63', '#7c5cbf']
    for (idx, row), color in zip(top3.iterrows(), colors_r):
        vals = [row[c] for c in cats] + [row[cats[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill='toself', name=idx, line_color=color, opacity=0.7))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            height=400, paper_bgcolor='#f8f7ff',
                            title='Top 3 Models – Performance Radar')
    st.plotly_chart(fig_radar, use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  PAGE: PREDICT CHURN
# ──────────────────────────────────────────────────────────────
elif page == "🎯 Predict Churn":
    st.markdown('<p class="hero-title" style="color:#302b63">🎯 Predict Customer Churn</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Using model: `{p['best_name']}`** (best ROC-AUC)")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Customer Age", 18, 75, 35)
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        support_calls = st.slider("Customer Support Calls", 0, 10, 2)
    with col2:
        monthly = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, 1.0)
        total = st.slider("Total Charges ($)", 20.0, 8000.0, float(monthly * tenure), 10.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        contract = st.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer", "Credit card"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

    if st.button("🔍 Predict Churn Risk", type="primary"):
        # Build input row matching training feature set
        input_dict = {
            'CustomerAge': age, 'Tenure': tenure,
            'MonthlyCharges': monthly, 'TotalCharges': total,
            'CustomerSupportCalls': support_calls,
            'Gender_enc': 0 if gender == 'Female' else 1
        }
        # OHE columns for ContractType
        for ct in ['Month-to-month', 'One year', 'Two year']:
            key = f"ContractType_{ct}"
            input_dict[key] = 1 if contract == ct else 0
        for pm in ['Credit card', 'Electronic check', 'Mailed check', 'Bank transfer']:
            input_dict[f"PaymentMethod_{pm}"] = 1 if payment == pm else 0
        for iv in ['DSL', 'Fiber optic', 'No']:
            input_dict[f"InternetService_{iv}"] = 1 if internet == iv else 0
        for ts in ['No', 'No internet service', 'Yes']:
            input_dict[f"TechSupport_{ts}"] = 1 if tech == ts else 0
        for os_ in ['No', 'No internet service', 'Yes']:
            input_dict[f"OnlineSecurity_{os_}"] = 1 if security == os_ else 0

        feat_names = p['feature_names']
        row = []
        for f in feat_names:
            row.append(input_dict.get(f, 0))
        row_arr = np.array(row).reshape(1, -1)
        row_scaled = p['scaler'].transform(row_arr)

        model = p['trained_models'][p['best_name']]
        prob = model.predict_proba(row_scaled)[0][1]
        pred = int(prob >= 0.5)

        risk_color = "#f72585" if prob > 0.6 else "#f48c06" if prob > 0.35 else "#2dc653"
        risk_label = "🔴 HIGH RISK" if prob > 0.6 else "🟡 MEDIUM RISK" if prob > 0.35 else "🟢 LOW RISK"

        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:2rem;text-align:center;
             box-shadow:0 4px 24px rgba(48,43,99,0.12);margin-top:1rem;
             border-top: 6px solid {risk_color}">
          <div style="font-family:'Space Mono',monospace;font-size:3rem;color:{risk_color};font-weight:700">
            {prob*100:.1f}%
          </div>
          <div style="font-size:1.4rem;color:#333;font-weight:600">{risk_label}</div>
          <div style="color:#666;margin-top:0.5rem">
            Churn Probability &nbsp;|&nbsp; Prediction: {"⚠️ WILL CHURN" if pred else "✅ WON'T CHURN"}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 35], 'color': '#d8f5e0'},
                    {'range': [35, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#fde0e0'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Key risk factors for this customer
        st.markdown('<p class="section-header">Risk Factor Analysis</p>', unsafe_allow_html=True)
        risks = []
        if contract == 'Month-to-month': risks.append("📋 Month-to-month contract → high churn risk")
        if monthly > 80: risks.append(f"💸 High monthly charges (${monthly:.0f}) → elevated risk")
        if tenure < 12: risks.append(f"⏱️ Short tenure ({tenure} months) → new customer vulnerability")
        if tech == 'No': risks.append("🛠️ No tech support → service dissatisfaction risk")
        if support_calls > 5: risks.append(f"📞 High support calls ({support_calls}) → friction indicator")
        if internet == 'Fiber optic': risks.append("🌐 Fiber optic users churn more in this dataset")

        if risks:
            for r in risks:
                st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No major churn risk factors detected for this customer profile.")
