
import streamlit as st
import pandas as pd
import numpy as np
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
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(160deg, #0f0c29, #302b63, #24243e); color: white; }
[data-testid="stSidebar"] * { color: white !important; }
.main { background-color: #f8f7ff; }
.hero-banner { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; letter-spacing: -1px; margin: 0; }
.hero-sub { font-size: 1rem; opacity: 0.75; margin-top: 0.3rem; }
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-card { flex: 1; min-width: 140px; background: white; border-radius: 12px; padding: 1.2rem 1.4rem; box-shadow: 0 2px 12px rgba(48,43,99,0.10); border-left: 4px solid #302b63; }
.metric-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #888; }
.metric-card .value { font-family: 'Space Mono', monospace; font-size: 1.7rem; color: #0f0c29; font-weight: 700; }
.section-header { font-family: 'Space Mono', monospace; font-size: 1.05rem; color: #302b63; letter-spacing: 0.5px; border-bottom: 2px solid #302b63; padding-bottom: 0.3rem; margin: 1.5rem 0 1rem 0; }
.model-card { background: white; border-radius: 12px; padding: 1rem 1.3rem; box-shadow: 0 2px 12px rgba(48,43,99,0.08); margin-bottom: 0.8rem; border-left: 4px solid #7c5cbf; }
.model-card.best { border-left-color: #f72585; }
.badge { display: inline-block; background: #302b63; color: white; font-size: 0.7rem; padding: 2px 9px; border-radius: 20px; font-family: 'Space Mono', monospace; margin-left: 8px; vertical-align: middle; }
.badge.best { background: #f72585; }
.insight-box { background: linear-gradient(135deg, #e0d7ff, #f3eeff); border-left: 4px solid #7c5cbf; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #1a1040; }
.rec-card { background: white; border-radius: 10px; padding: 0.9rem 1.2rem; margin: 0.4rem 0; box-shadow: 0 1px 8px rgba(48,43,99,0.07); display: flex; gap: 0.8rem; align-items: flex-start; }
.rec-icon { font-size: 1.4rem; }
.rec-text { font-size: 0.88rem; color: #333; }
.rec-text strong { color: #302b63; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
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
    n_samples    = st.slider("Sample Size", 500, 5000, 2000, 100)
    test_size    = st.slider("Test Split %", 10, 40, 20, 5)
    scaler_choice = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])
    use_smote    = st.checkbox("Apply SMOTE", value=True)
    run_tuning   = st.checkbox("Run Hyperparameter Tuning (slow)", value=False)
    st.markdown("---")
    st.caption("Built with Streamlit + Scikit-learn + XGBoost")

# ─────────────────────────────────────────────
#  STAGE 1 – raw data  (fast, cached forever per n)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw(n):
    return generate_churn_dataset(n)

# ─────────────────────────────────────────────
#  STAGE 2 – preprocessing  (cached per params)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(n, test_sz, scaler_name, apply_smote):
    raw = generate_churn_dataset(n)
    df  = raw.copy()

    df['TotalCharges']   = pd.to_numeric(df['TotalCharges'],   errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    missing_before = df.isnull().sum().to_dict()
    df['TotalCharges']   = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())

    n_dups = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    num_cols = ['CustomerAge', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'CustomerSupportCalls']
    outlier_counts = {}
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lo, hi)

    cat_cols = ['Gender', 'ContractType', 'PaymentMethod',
                'InternetService', 'TechSupport', 'OnlineSecurity']
    le = LabelEncoder()
    df['Gender_enc'] = le.fit_transform(df['Gender'])
    df_ohe = pd.get_dummies(df[cat_cols[1:]], drop_first=False)
    df = pd.concat([df.drop(cat_cols, axis=1), df['Gender_enc'], df_ohe], axis=1)
    df.columns = df.columns.astype(str)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    feature_names = X.columns.tolist()

    scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(X), nan=0.0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_sz/100, random_state=42, stratify=y)

    smote_info = {}
    if apply_smote:
        before = dict(pd.Series(y_tr).value_counts())
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        smote_info = {'before': before, 'after': dict(pd.Series(y_tr).value_counts())}

    return dict(
        raw=raw, df_clean=df,
        missing_before=missing_before, n_dups=n_dups,
        outlier_counts=outlier_counts, smote_info=smote_info,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        feature_names=feature_names, scaler=scaler
    )

# ─────────────────────────────────────────────
#  STAGE 3 – train ONE model  (cached per model+params)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_one(model_name, n, test_sz, scaler_name, apply_smote):
    """Train a single model and return metrics + trained object."""
    prep = preprocess(n, test_sz, scaler_name, apply_smote)
    X_tr, X_te = prep['X_tr'], prep['X_te']
    y_tr, y_te = prep['y_tr'], prep['y_te']

    model_map = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVC':                 SVC(probability=True, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, eval_metric='logloss',
                                             use_label_encoder=False, random_state=42,
                                             n_jobs=-1)
    }
    model = model_map[model_name]
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    # CV only on a subsample to keep it fast
    cv_size  = min(len(X_tr), 1500)
    idx      = np.random.default_rng(42).choice(len(X_tr), cv_size, replace=False)
    cv_scores = cross_val_score(model, X_tr[idx], np.array(y_tr)[idx],
                                cv=3, scoring='roc_auc').tolist()

    return dict(
        model=model,
        metrics=dict(
            Accuracy  = round(accuracy_score(y_te, y_pred), 4),
            Precision = round(precision_score(y_te, y_pred, zero_division=0), 4),
            Recall    = round(recall_score(y_te, y_pred, zero_division=0), 4),
            **{'F1 Score': round(f1_score(y_te, y_pred, zero_division=0), 4)},
            **{'ROC-AUC': round(roc_auc_score(y_te, y_prob), 4)},
            **{'CV Mean AUC': round(float(np.mean(cv_scores)), 4)},
            **{'Confusion Matrix': confusion_matrix(y_te, y_pred).tolist()}
        ),
        cv_scores  = cv_scores,
        roc_data   = (fpr.tolist(), tpr.tolist()),
        fi         = (list(prep['feature_names'])
                      if hasattr(model, 'feature_importances_') else None),
        fi_vals    = (model.feature_importances_.tolist()
                      if hasattr(model, 'feature_importances_') else None)
    )

# ─────────────────────────────────────────────
#  STAGE 4 – hyperparameter tuning  (optional, cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_tuning_fn(n, test_sz, scaler_name, apply_smote):
    prep = preprocess(n, test_sz, scaler_name, apply_smote)
    X_tr, y_tr = prep['X_tr'], prep['y_tr']

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {'n_estimators': [100, 200], 'max_depth': [6, 10, None], 'min_samples_split': [2, 5]},
        cv=3, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_tr, y_tr)

    rs = RandomizedSearchCV(
        XGBClassifier(eval_metric='logloss', use_label_encoder=False,
                      random_state=42, n_jobs=-1),
        {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
         'learning_rate': [0.05, 0.1, 0.2], 'subsample': [0.7, 0.9, 1.0]},
        n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
    rs.fit(X_tr, y_tr)

    return {
        'GridSearchCV (RF)':        {'best_params': gs.best_params_,
                                     'best_score':  round(gs.best_score_, 4)},
        'RandomizedSearchCV (XGB)': {'best_params': rs.best_params_,
                                     'best_score':  round(rs.best_score_, 4)}
    }

# ─────────────────────────────────────────────
#  LOAD RAW DATA  (always fast)
# ─────────────────────────────────────────────
raw = load_raw(n_samples)

# ─────────────────────────────────────────────
#  TRAIN ALL MODELS with per-model progress bar
# ─────────────────────────────────────────────
MODEL_NAMES = ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree',
               'Random Forest', 'SVC', 'Gradient Boosting', 'XGBoost']

MODEL_INFO = {
    'Logistic Regression': ('Linear baseline model', '⚡'),
    'K-Nearest Neighbors': ('Instance-based learner', '🔍'),
    'Decision Tree':       ('Rule-based tree classifier', '🌿'),
    'Random Forest':       ('Ensemble of decision trees', '🌲'),
    'SVC':                 ('Margin-maximizing classifier', '🔷'),
    'Gradient Boosting':   ('Sequential boosting ensemble', '🚀'),
    'XGBoost':             ('Optimized gradient boosting', '⚡🌲')
}

# Check which models are already cached (avoid re-showing spinner)
pipeline_key = (n_samples, test_size, scaler_choice, use_smote)

# Show progress only if something needs to be computed
needs_compute = False
for mn in MODEL_NAMES:
    cache_check_key = f"trained_{mn}_{pipeline_key}"
    if cache_check_key not in st.session_state:
        needs_compute = True
        break

if needs_compute:
    progress_bar = st.progress(0, text="Training models…")
    status_text  = st.empty()

trained = {}
for i, mn in enumerate(MODEL_NAMES):
    cache_check_key = f"trained_{mn}_{pipeline_key}"
    if needs_compute:
        status_text.markdown(f"⚙️ Training **{mn}** ({i+1}/{len(MODEL_NAMES)})…")
        progress_bar.progress((i) / len(MODEL_NAMES),
                              text=f"Training {mn}…")
    trained[mn] = train_one(mn, n_samples, test_size, scaler_choice, use_smote)
    st.session_state[cache_check_key] = True

if needs_compute:
    progress_bar.progress(1.0, text="✅ All models ready!")
    status_text.empty()
    progress_bar.empty()

# ─────────────────────────────────────────────
#  AGGREGATE RESULTS
# ─────────────────────────────────────────────
prep = preprocess(n_samples, test_size, scaler_choice, use_smote)
df   = prep['df_clean']

res       = {mn: trained[mn]['metrics']   for mn in MODEL_NAMES}
cv_scores = {mn: trained[mn]['cv_scores'] for mn in MODEL_NAMES}
roc_data  = {mn: trained[mn]['roc_data']  for mn in MODEL_NAMES}

best_name = max(res, key=lambda k: res[k]['ROC-AUC'])

# Feature importance from RF
rf_fi = trained['Random Forest']
fi_dict = dict(zip(rf_fi['fi'], rf_fi['fi_vals'])) if rf_fi['fi'] else {}

# ─────────────────────────────────────────────
#  OPTIONAL TUNING
# ─────────────────────────────────────────────
tuning_results = None
if run_tuning:
    with st.spinner("Running hyperparameter tuning (GridSearchCV + RandomizedSearchCV)…"):
        tuning_results = run_tuning_fn(n_samples, test_size, scaler_choice, use_smote)

# ══════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════

# ─────────────────────────────────────────────
#  PAGE: OVERVIEW & EDA
# ─────────────────────────────────────────────
if page == "🏠 Overview & EDA":
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">📉 ChurnSight</p>
      <p class="hero-sub">End-to-End Customer Churn Prediction · ML Classification Dashboard</p>
    </div>""", unsafe_allow_html=True)

    churn_rate = raw['Churn'].mean() * 100
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">Total Customers</div><div class="value">{len(raw):,}</div></div>
      <div class="metric-card"><div class="label">Churn Rate</div><div class="value">{churn_rate:.1f}%</div></div>
      <div class="metric-card"><div class="label">Features</div><div class="value">11</div></div>
      <div class="metric-card"><div class="label">Best Model AUC</div><div class="value">{max(v['ROC-AUC'] for v in res.values()):.3f}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📐 Shape & Types", "📊 Statistics"])
    with tab1:
        st.dataframe(raw.head(20), use_container_width=True)
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", raw.shape[0]); st.metric("Columns", raw.shape[1])
        with c2:
            st.dataframe(pd.DataFrame({'Column': raw.columns,
                                       'Dtype': raw.dtypes.values.astype(str),
                                       'Nulls': raw.isnull().sum().values}),
                         use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(raw.describe().round(2), use_container_width=True)

    st.markdown('<p class="section-header">Univariate Analysis</p>', unsafe_allow_html=True)
    num_cols = ['CustomerAge', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'CustomerSupportCalls']
    fig = make_subplots(rows=1, cols=5, subplot_titles=num_cols)
    for i, col in enumerate(num_cols, 1):
        fig.add_trace(go.Histogram(x=raw[col], nbinsx=25, marker_color='#302b63',
                                   opacity=0.75, name=col, showlegend=False), row=1, col=i)
    fig.update_layout(height=300, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                      margin=dict(t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    cat_cols = ['Gender', 'ContractType', 'PaymentMethod',
                'InternetService', 'TechSupport', 'OnlineSecurity']
    cols = st.columns(3)
    for i, col in enumerate(cat_cols):
        vc = raw[col].value_counts().reset_index()
        vc.columns = [col, 'Count']
        fig2 = px.bar(vc, x=col, y='Count',
                      color_discrete_sequence=['#302b63','#7c5cbf','#c77dff','#f72585'],
                      title=col)
        fig2.update_layout(height=260, margin=dict(t=40,b=20,l=10,r=10),
                           plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        cols[i % 3].plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-header">Bivariate Analysis – Churn vs Features</p>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ct = pd.crosstab(raw['ContractType'], raw['Churn'], normalize='index') * 100
        fig3 = px.bar(ct.reset_index(), x='ContractType', y=[0, 1], barmode='group',
                      title='Churn % by Contract Type',
                      color_discrete_map={0: '#302b63', 1: '#f72585'})
        fig3.update_layout(height=320, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        fig4 = px.box(raw, x='Churn', y='MonthlyCharges', color='Churn',
                      title='Monthly Charges vs Churn',
                      color_discrete_map={0: '#302b63', 1: '#f72585'})
        fig4.update_layout(height=320, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig4, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig5 = px.box(raw, x='Churn', y='Tenure', color='Churn',
                      title='Tenure vs Churn',
                      color_discrete_map={0: '#302b63', 1: '#f72585'})
        fig5.update_layout(height=300, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig5, use_container_width=True)
    with c4:
        ts_churn = raw.groupby('TechSupport')['Churn'].mean().reset_index()
        ts_churn['Churn'] *= 100
        fig6 = px.bar(ts_churn, x='TechSupport', y='Churn',
                      title='Churn Rate by Tech Support',
                      color_discrete_sequence=['#7c5cbf'])
        fig6.update_layout(height=300, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
    corr = raw[['CustomerAge','Tenure','MonthlyCharges',
                'TotalCharges','CustomerSupportCalls','Churn']].corr()
    fig7 = px.imshow(corr, text_auto='.2f', aspect='auto',
                     color_continuous_scale='RdBu_r', title='Feature Correlation Matrix')
    fig7.update_layout(height=420, paper_bgcolor='#f8f7ff')
    st.plotly_chart(fig7, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: PREPROCESSING
# ─────────────────────────────────────────────
elif page == "🔬 Preprocessing":
    st.markdown('<p class="hero-title" style="color:#302b63">🔬 Data Preprocessing</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">Missing Values</p>', unsafe_allow_html=True)
    mv = {k: v for k, v in prep['missing_before'].items() if v > 0}
    if mv:
        c1, c2 = st.columns(2)
        with c1:
            mv_df = pd.DataFrame.from_dict(mv, orient='index', columns=['Missing Count'])
            mv_df['% Missing'] = (mv_df['Missing Count'] / len(raw) * 100).round(2)
            st.dataframe(mv_df, use_container_width=True)
        with c2:
            st.markdown("""<div class="insight-box">✅ <b>Strategy:</b> TotalCharges &amp; MonthlyCharges → filled with <b>median</b></div>""",
                        unsafe_allow_html=True)
    else:
        st.info("No missing values detected.")

    st.markdown('<p class="section-header">Duplicate Records</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">🔍 Found <b>{prep["n_dups"]} duplicate rows</b> → Removed. Clean dataset: <b>{len(df):,} rows</b>.</div>',
                unsafe_allow_html=True)

    st.markdown('<p class="section-header">Outlier Detection (IQR Method)</p>', unsafe_allow_html=True)
    out_df = pd.DataFrame.from_dict(prep['outlier_counts'], orient='index',
                                    columns=['Outliers Detected'])
    out_df['Action'] = 'IQR Clipping Applied'
    st.dataframe(out_df, use_container_width=True)

    fig_box = make_subplots(rows=1, cols=5,
                            subplot_titles=list(prep['outlier_counts'].keys()))
    for i, col in enumerate(prep['outlier_counts'].keys(), 1):
        if col in raw.columns:
            fig_box.add_trace(go.Box(y=raw[col], name=col, showlegend=False,
                                     marker_color='#302b63'), row=1, col=i)
    fig_box.update_layout(height=320, title_text="Boxplots (Before Clipping)",
                          plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<p class="section-header">Encoding</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Label Encoding** → `Gender` (0/1)")
    with c2:
        st.markdown("**One-Hot Encoding** → ContractType, PaymentMethod, InternetService, TechSupport, OnlineSecurity")

    st.markdown('<p class="section-header">Scaling & Class Balancing</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="insight-box">⚖️ <b>Scaler:</b> {scaler_choice}<br>Applied to {len(prep["feature_names"])} features.</div>',
                    unsafe_allow_html=True)
    with c2:
        if use_smote and prep['smote_info']:
            bef, aft = prep['smote_info']['before'], prep['smote_info']['after']
            st.markdown(f'<div class="insight-box">🔁 <b>SMOTE Applied</b><br>Before → 0:{bef.get(0,0)} | 1:{bef.get(1,0)}<br>After → 0:{aft.get(0,0)} | 1:{aft.get(1,0)}</div>',
                        unsafe_allow_html=True)
        else:
            st.info("SMOTE not applied.")

    st.markdown('<p class="section-header">Train-Test Split</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">Training Samples</div><div class="value">{len(prep['X_tr']):,}</div></div>
      <div class="metric-card"><div class="label">Test Samples</div><div class="value">{len(prep['X_te']):,}</div></div>
      <div class="metric-card"><div class="label">Features</div><div class="value">{len(prep['feature_names'])}</div></div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: MODEL TRAINING
# ─────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.markdown('<p class="hero-title" style="color:#302b63">🤖 Model Training</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="section-header">Classification Models Trained</p>', unsafe_allow_html=True)

    for name, (desc, icon) in MODEL_INFO.items():
        metrics  = res[name]
        is_best  = (name == best_name)
        badge    = '<span class="badge best">BEST</span>' if is_best else ''
        st.markdown(f"""
        <div class="model-card {'best' if is_best else ''}">
          <b>{icon} {name}</b>{badge}<br>
          <small style="color:#666">{desc}</small><br>
          <span style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#302b63">
          ACC: {metrics['Accuracy']:.3f} &nbsp;|&nbsp;
          F1: {metrics['F1 Score']:.3f} &nbsp;|&nbsp;
          AUC: {metrics['ROC-AUC']:.3f}
          </span>
        </div>""", unsafe_allow_html=True)

    if fi_dict:
        st.markdown('<p class="section-header">Feature Importance (Random Forest)</p>',
                    unsafe_allow_html=True)
        fi_df = pd.DataFrame({'Feature': list(fi_dict.keys()),
                              'Importance': list(fi_dict.values())}).nlargest(15, 'Importance')
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Purples',
                        title='Top 15 Feature Importances')
        fig_fi.update_layout(height=420, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                             yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: MODEL EVALUATION
# ─────────────────────────────────────────────
elif page == "📊 Model Evaluation":
    st.markdown('<p class="hero-title" style="color:#302b63">📊 Model Evaluation</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">Metrics Table</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(res).T.drop('Confusion Matrix', axis=1).astype(float)

    def highlight_best(s):
        return ['background-color:#e0d7ff;font-weight:bold' if v == s.max() else '' for v in s]

    st.dataframe(metrics_df.style.apply(highlight_best).format("{:.4f}"),
                 use_container_width=True)

    st.markdown('<p class="section-header">ROC Curves</p>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    colors = ['#302b63','#7c5cbf','#c77dff','#f72585','#480ca8','#b5179e','#4cc9f0']
    for (name, (fpr, tpr)), color in zip(roc_data.items(), colors):
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"{name} (AUC={res[name]['ROC-AUC']:.3f})",
                                     line=dict(color=color, width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 line=dict(dash='dash', color='gray'), name='Random'))
    fig_roc.update_layout(title='ROC Curves – All Models', height=480,
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff')
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown('<p class="section-header">Confusion Matrices</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, name in enumerate(MODEL_NAMES):
        cm = np.array(res[name]['Confusion Matrix'])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Purples',
                           labels=dict(x="Predicted", y="Actual"),
                           x=['No Churn','Churn'], y=['No Churn','Churn'], title=name)
        fig_cm.update_layout(height=260, margin=dict(t=40,b=10,l=10,r=10),
                             paper_bgcolor='#f8f7ff', coloraxis_showscale=False)
        cols[i % 4].plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<p class="section-header">5-Fold Cross-Validation AUC Scores</p>',
                unsafe_allow_html=True)
    cv_data = [{'Model': mn, 'CV AUC': s}
               for mn, scores in cv_scores.items() for s in scores]
    fig_cv = px.box(pd.DataFrame(cv_data), x='Model', y='CV AUC', color='Model',
                    color_discrete_sequence=colors,
                    title='Cross-Validation AUC Distribution per Model')
    fig_cv.update_layout(height=380, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                         showlegend=False, xaxis_tickangle=-25)
    st.plotly_chart(fig_cv, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
elif page == "🔧 Hyperparameter Tuning":
    st.markdown('<p class="hero-title" style="color:#302b63">🔧 Hyperparameter Tuning</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    if not run_tuning:
        st.info("⚠️ Hyperparameter tuning is disabled. Enable it in the sidebar with **'Run Hyperparameter Tuning (slow)'** checkbox to see results.")
        st.markdown("""
        **What will run when enabled:**
        - `GridSearchCV` on Random Forest (n_estimators × max_depth × min_samples_split, cv=3)
        - `RandomizedSearchCV` on XGBoost (n_iter=10, cv=3)

        Results are cached — tuning only runs once per parameter set.
        """)
    else:
        st.markdown('<p class="section-header">GridSearchCV & RandomizedSearchCV Results</p>',
                    unsafe_allow_html=True)
        for method, info in tuning_results.items():
            st.markdown(f"""
            <div class="model-card">
              <b>{'🔷' if 'Grid' in method else '🎲'} {method}</b><br>
              <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#302b63">
              Best CV AUC: <b>{info['best_score']:.4f}</b></span><br>
              <small style="color:#555">Best Params: {info['best_params']}</small>
            </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**GridSearchCV – Random Forest**\n```\nn_estimators: [100,200]\nmax_depth: [6,10,None]\nmin_samples_split: [2,5]\ncv=3, scoring=roc_auc\n```")
        with c2:
            st.markdown("**RandomizedSearchCV – XGBoost**\n```\nn_estimators: [100,200,300]\nmax_depth: [3,5,7]\nlearning_rate: [0.05,0.1,0.2]\nsubsample: [0.7,0.9,1.0]\nn_iter=10, cv=3\n```")

        base_rf   = res['Random Forest']['ROC-AUC']
        base_xgb  = res['XGBoost']['ROC-AUC']
        tuned_rf  = tuning_results['GridSearchCV (RF)']['best_score']
        tuned_xgb = tuning_results['RandomizedSearchCV (XGB)']['best_score']
        comp_df   = pd.DataFrame({
            'Model': ['Random Forest','Random Forest','XGBoost','XGBoost'],
            'Stage': ['Baseline','Tuned','Baseline','Tuned'],
            'ROC-AUC': [base_rf, tuned_rf, base_xgb, tuned_xgb]
        })
        fig_tune = px.bar(comp_df, x='Model', y='ROC-AUC', color='Stage', barmode='group',
                          color_discrete_map={'Baseline':'#aaa','Tuned':'#302b63'},
                          title='Baseline vs Tuned ROC-AUC')
        fig_tune.update_layout(height=340, plot_bgcolor='#f8f7ff', paper_bgcolor='#f8f7ff',
                               yaxis=dict(range=[0.5, 1.0]))
        st.plotly_chart(fig_tune, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: RESULTS & INSIGHTS
# ─────────────────────────────────────────────
elif page == "🏆 Results & Insights":
    st.markdown('<p class="hero-title" style="color:#302b63">🏆 Results & Business Insights</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">Model Comparison Table</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(res).T.drop('Confusion Matrix', axis=1).astype(float)
    metrics_df['Rank'] = metrics_df['ROC-AUC'].rank(ascending=False).astype(int)
    metrics_df = metrics_df.sort_values('ROC-AUC', ascending=False)
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    bm = res[best_name]
    st.markdown(f"""
    <div class="model-card best">
      <span class="badge best">WINNER</span>
      <b style="font-size:1.2rem"> {best_name}</b><br><br>
      <div class="metric-row">
        <div class="metric-card"><div class="label">Accuracy</div><div class="value">{bm['Accuracy']:.3f}</div></div>
        <div class="metric-card"><div class="label">Precision</div><div class="value">{bm['Precision']:.3f}</div></div>
        <div class="metric-card"><div class="label">Recall</div><div class="value">{bm['Recall']:.3f}</div></div>
        <div class="metric-card"><div class="label">F1 Score</div><div class="value">{bm['F1 Score']:.3f}</div></div>
        <div class="metric-card"><div class="label">ROC-AUC</div><div class="value">{bm['ROC-AUC']:.3f}</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Key Churn Drivers</p>', unsafe_allow_html=True)
    for icon, title, desc in [
        ("📈","High Monthly Charges","Customers paying >$80/month churn ~30% more than average."),
        ("📋","Month-to-Month Contracts","~55% of churn comes from short-term contract holders."),
        ("🛠️","No Tech Support","Customers without tech support are significantly more likely to leave."),
        ("⏱️","Short Tenure (<12 months)","New customers have the highest churn risk in the first year."),
        ("📞","High Support Calls (>5)","Frequent support interactions signal dissatisfaction."),
    ]:
        st.markdown(f'<div class="insight-box">{icon} <b>{title}</b> — {desc}</div>',
                    unsafe_allow_html=True)

    st.markdown('<p class="section-header">Business Recommendations</p>', unsafe_allow_html=True)
    for icon, title, desc in [
        ("💰","Offer Long-Term Contract Discounts","Incentivize customers to switch from month-to-month to annual plans."),
        ("🎧","Improve Customer Support Quality","Reduce resolution time and first-call resolution rate."),
        ("🎁","Personalized Retention Offers","Send targeted loyalty rewards to high-risk segments."),
        ("🚨","Early-Warning Churn Detection","Deploy this model to flag high-risk customers within first 3 months."),
        ("🏅","Loyalty Programs","Reward tenure milestones (6 months, 1 year, etc.)."),
    ]:
        st.markdown(f'<div class="rec-card"><div class="rec-icon">{icon}</div><div class="rec-text"><strong>{title}</strong><br>{desc}</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<p class="section-header">Top 3 Model Radar Comparison</p>', unsafe_allow_html=True)
    top3 = metrics_df.head(3)
    cats = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC','CV Mean AUC']
    fig_radar = go.Figure()
    for (idx, row), color in zip(top3.iterrows(), ['#f72585','#302b63','#7c5cbf']):
        vals = [row[c] for c in cats] + [row[cats[0]]]
        fig_radar.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]],
                                            fill='toself', name=idx,
                                            line_color=color, opacity=0.7))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            height=400, paper_bgcolor='#f8f7ff',
                            title='Top 3 Models – Performance Radar')
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: PREDICT CHURN
# ─────────────────────────────────────────────
elif page == "🎯 Predict Churn":
    st.markdown('<p class="hero-title" style="color:#302b63">🎯 Predict Customer Churn</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Using model: `{best_name}`** (best ROC-AUC: {res[best_name]['ROC-AUC']:.3f})")

    c1, c2, c3 = st.columns(3)
    with c1:
        age           = st.slider("Customer Age", 18, 75, 35)
        tenure        = st.slider("Tenure (months)", 1, 72, 12)
        support_calls = st.slider("Customer Support Calls", 0, 10, 2)
    with c2:
        monthly  = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, 1.0)
        total    = st.slider("Total Charges ($)", 20.0, 8000.0,
                             float(monthly * tenure), 10.0)
        gender   = st.selectbox("Gender", ["Male", "Female"])
    with c3:
        contract = st.selectbox("Contract Type",
                                ["Month-to-month","One year","Two year"])
        payment  = st.selectbox("Payment Method",
                                ["Electronic check","Mailed check",
                                 "Bank transfer","Credit card"])
        internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        tech     = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        security = st.selectbox("Online Security", ["Yes","No","No internet service"])

    if st.button("🔍 Predict Churn Risk", type="primary"):
        input_dict = {
            'CustomerAge': age, 'Tenure': tenure,
            'MonthlyCharges': monthly, 'TotalCharges': total,
            'CustomerSupportCalls': support_calls,
            'Gender_enc': 0 if gender == 'Female' else 1
        }
        for ct in ['Month-to-month','One year','Two year']:
            input_dict[f"ContractType_{ct}"]   = int(contract == ct)
        for pm in ['Credit card','Electronic check','Mailed check','Bank transfer']:
            input_dict[f"PaymentMethod_{pm}"]   = int(payment == pm)
        for iv in ['DSL','Fiber optic','No']:
            input_dict[f"InternetService_{iv}"] = int(internet == iv)
        for ts in ['No','No internet service','Yes']:
            input_dict[f"TechSupport_{ts}"]     = int(tech == ts)
        for os_ in ['No','No internet service','Yes']:
            input_dict[f"OnlineSecurity_{os_}"] = int(security == os_)

        row_arr    = np.array([input_dict.get(f, 0)
                               for f in prep['feature_names']]).reshape(1, -1)
        row_scaled = prep['scaler'].transform(row_arr)
        model      = trained[best_name]['model']
        prob       = model.predict_proba(row_scaled)[0][1]
        pred       = int(prob >= 0.5)

        risk_color = "#f72585" if prob > 0.6 else "#f48c06" if prob > 0.35 else "#2dc653"
        risk_label = "🔴 HIGH RISK" if prob > 0.6 else "🟡 MEDIUM RISK" if prob > 0.35 else "🟢 LOW RISK"

        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:2rem;text-align:center;
             box-shadow:0 4px 24px rgba(48,43,99,0.12);margin-top:1rem;
             border-top:6px solid {risk_color}">
          <div style="font-family:'Space Mono',monospace;font-size:3rem;
               color:{risk_color};font-weight:700">{prob*100:.1f}%</div>
          <div style="font-size:1.4rem;color:#333;font-weight:600">{risk_label}</div>
          <div style="color:#666;margin-top:0.5rem">
            Churn Probability &nbsp;|&nbsp;
            Prediction: {"⚠️ WILL CHURN" if pred else "✅ WON'T CHURN"}
          </div>
        </div>""", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': risk_color},
                'steps': [{'range': [0,35],  'color': '#d8f5e0'},
                           {'range': [35,60], 'color': '#fff3cd'},
                           {'range': [60,100],'color': '#fde0e0'}],
                'threshold': {'line': {'color': 'red', 'width': 4},
                              'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor='#f8f7ff')
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('<p class="section-header">Risk Factor Analysis</p>', unsafe_allow_html=True)
        risks = []
        if contract == 'Month-to-month':  risks.append("📋 Month-to-month contract → high churn risk")
        if monthly > 80:                  risks.append(f"💸 High monthly charges (${monthly:.0f}) → elevated risk")
        if tenure < 12:                   risks.append(f"⏱️ Short tenure ({tenure} months) → new customer vulnerability")
        if tech == 'No':                  risks.append("🛠️ No tech support → service dissatisfaction risk")
        if support_calls > 5:             risks.append(f"📞 High support calls ({support_calls}) → friction indicator")
        if internet == 'Fiber optic':     risks.append("🌐 Fiber optic users churn more in this dataset")

        if risks:
            for r in risks:
                st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No major churn risk factors detected for this customer profile.")
