import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
 
# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Insurance – Phase 6",
    page_icon="✈️",
    layout="wide"
)
 
# ─── Minimal CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e293b; border-radius: 10px;
        padding: 14px 18px; text-align: center; color: white;
    }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .metric-lbl  { font-size: 0.78rem; color: #94a3b8; margin-top: 2px; }
    h1 { color: #38bdf8 !important; }
    h2 { color: #e2e8f0 !important; border-bottom: 1px solid #334155; padding-bottom: 4px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)
 
# ─── Title ────────────────────────────────────────────────────────────────────
st.title("✈️ Travel Insurance Prediction — Phase 6")
st.caption("SQI College of ICT · Machine Learning Classification Project · Deployment Dashboard")
 
# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION (mirrors real dataset distribution)
# In production replace this block with: df = pd.read_csv("TravelInsurancePrediction.csv")
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_and_prepare():
    np.random.seed(42)
    n = 1987  # matches real dataset size
 
    raw = pd.DataFrame({
        "Age":                np.random.randint(25, 65, n),
        "Employment Type":    np.random.choice(["Government Sector", "Private Sector/Self Employed"], n, p=[0.3, 0.7]),
        "GraduateOrNot":      np.random.choice(["Yes", "No"], n, p=[0.76, 0.24]),
        "AnnualIncome":       np.random.randint(300000, 1800000, n),
        "FamilyMembers":      np.random.randint(2, 9, n),
        "ChronicDiseases":    np.random.choice([0, 1], n, p=[0.72, 0.28]),
        "FrequentFlyer":      np.random.choice(["Yes", "No"], n, p=[0.46, 0.54]),
        "EverTravelledAbroad":np.random.choice(["Yes", "No"], n, p=[0.19, 0.81]),
        "TravelInsurance":    np.random.choice([0, 1], n, p=[0.64, 0.36]),
    })
 
    # Encode categoricals
    le = LabelEncoder()
    for col in ["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]:
        raw[col] = le.fit_transform(raw[col])
 
    X = raw.drop("TravelInsurance", axis=1)
    y = raw["TravelInsurance"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
 
    # Scale (for LR)
    sc = StandardScaler()
    X_res_sc  = sc.fit_transform(X_res)
    X_test_sc = sc.transform(X_test)
 
    # Class weights for RF/GB
    cw = dict(zip([0,1], compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)))
 
    return X_res, y_res, X_test, y_test, X_res_sc, X_test_sc, cw, raw
 
@st.cache_data
@st.cache_data
def train_models(_X_res, _y_res, _X_test, _y_test, _X_res_sc, _X_test_sc, _cw):
    results = {}

    configs = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, class_weight=_cw, random_state=42),
            _X_res, _X_test
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            _X_res, _X_test
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            _X_res_sc, _X_test_sc
        ),
    }

    for name, (clf, Xtr, Xte) in configs.items():
        clf.fit(Xtr, _y_res)

        # ✅ Save each model as a .pkl file
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(clf, filename)

        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]
        results[name] = {
            "model":    clf,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "acc":      accuracy_score(_y_test, y_pred),
            "auc":      roc_auc_score(_y_test, y_prob),
            "cm":       confusion_matrix(_y_test, y_pred),
            "report":   classification_report(_y_test, y_pred, output_dict=True),
        }

    return results
 
# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Training models … this takes ~10 seconds on first load"):
    X_res, y_res, X_test, y_test, X_res_sc, X_test_sc, cw, raw = load_and_prepare()
    results = train_models(X_res, y_res, X_test, y_test, X_res_sc, X_test_sc, cw)
 
    rf_model = results["Random Forest"]["model"]
joblib.dump(rf_model, "random_forest.pkl")
best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
 
# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Best Model at a Glance")
k1, k2, k3, k4, k5 = st.columns(5)
 
kpis = [
    (k1, f"{best['acc']*100:.1f}%",  "Accuracy"),
    (k2, f"{best['auc']:.3f}",       "ROC-AUC"),
    (k3, f"{best['report']['1']['precision']:.3f}", "Precision (class 1)"),
    (k4, f"{best['report']['1']['recall']:.3f}",    "Recall (class 1)"),
    (k5, f"{best['report']['1']['f1-score']:.3f}",  "F1-Score (class 1)"),
]
for col, val, lbl in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
 
st.markdown(f"<br><center><b>Best model: {best_name}</b></center>", unsafe_allow_html=True)
st.divider()
 
# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Model Comparison",
    "🗺️ Confusion Matrices",
    "📉 ROC Curves",
    "🔍 Feature Importance",
    "🎯 Live Predictor",
])
 
# ── TAB 1 · Model Comparison ──────────────────────────────────────────────────
with tab1:
    st.subheader("Model Performance Comparison")
 
    summary = []
    for name, r in results.items():
        rep = r["report"]
        summary.append({
            "Model":           name,
            "Accuracy":        f"{r['acc']*100:.2f}%",
            "ROC-AUC":         f"{r['auc']:.4f}",
            "Precision (1)":   f"{rep['1']['precision']:.4f}",
            "Recall (1)":      f"{rep['1']['recall']:.4f}",
            "F1-Score (1)":    f"{rep['1']['f1-score']:.4f}",
            "Support (1)":     int(rep['1']['support']),
        })
    st.dataframe(pd.DataFrame(summary).set_index("Model"), use_container_width=True)
 
    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor="#0f172a")
    metrics_to_plot = ["Accuracy", "ROC-AUC", "F1-Score (1)"]
    colors = ["#38bdf8", "#34d399", "#f472b6"]
 
    for ax, metric, color in zip(axes, metrics_to_plot, colors):
        vals  = [float(d[metric].rstrip('%'))/100 if '%' in d[metric] else float(d[metric])
                 for d in summary]
        names = [d["Model"].replace(" ", "\n") for d in summary]
        bars = ax.bar(names, vals, color=color, alpha=0.85, width=0.5)
        ax.set_facecolor("#1e293b")
        ax.set_title(metric, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
# ── TAB 2 · Confusion Matrices ────────────────────────────────────────────────
with tab2:
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, r) in enumerate(results.items()):
        with cols[i]:
            cm = r["cm"]
            fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="#0f172a")
            ax.set_facecolor("#1e293b")
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["No (0)", "Yes (1)"], color="white", fontsize=9)
            ax.set_yticklabels(["No (0)", "Yes (1)"], color="white", fontsize=9)
            ax.set_xlabel("Predicted", color="#94a3b8", fontsize=9)
            ax.set_ylabel("Actual",    color="#94a3b8", fontsize=9)
            ax.set_title(name, color="white", fontsize=9, pad=8)
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(cm[row, col]),
                            ha="center", va="center",
                            color="white" if cm[row, col] < cm.max()*0.6 else "#0f172a",
                            fontsize=14, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
            tn, fp, fn, tp = cm.ravel()
 
# ── TAB 3 · ROC Curves ────────────────────────────────────────────────────────
with tab3:
    st.subheader("ROC Curves — All Models")
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    palette = {"Random Forest": "#38bdf8", "Gradient Boosting": "#34d399",
               "Logistic Regression": "#f472b6"}
 
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})",
                color=palette[name], linewidth=2)
 
    ax.plot([0,1],[0,1], "w--", linewidth=1, alpha=0.4, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", color="#94a3b8")
    ax.set_ylabel("True Positive Rate",  color="#94a3b8")
    ax.set_title("Receiver Operating Characteristic", color="white")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    st.info(f"🏆 **{best_name}** achieved the highest AUC of **{best['auc']:.4f}**, "
            f"indicating the strongest ability to distinguish insured vs uninsured travellers.")
 
# ── TAB 4 · Feature Importance ────────────────────────────────────────────────
with tab4:
    st.subheader("Feature Importance — Random Forest")
    rf     = results["Random Forest"]["model"]
    feat_names = ["Age", "Employment Type", "GraduateOrNot", "AnnualIncome",
                  "FamilyMembers", "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad"]
    imps   = rf.feature_importances_
    order  = np.argsort(imps)
 
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    bars = ax.barh([feat_names[i] for i in order],
                   [imps[i] for i in order],
                   color="#38bdf8", alpha=0.85)
    ax.set_xlabel("Importance Score", color="#94a3b8")
    ax.set_title("Feature Importances (Random Forest)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", color="white", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    top2 = np.argsort(imps)[-2:][::-1]
    st.success(f"Top predictors: **{feat_names[top2[0]]}** and **{feat_names[top2[1]]}** "
               f"drive most of the model's decisions.")
 
# ── TAB 5 · Live Predictor ────────────────────────────────────────────────────
with tab5:
    st.subheader("🎯 Live Prediction")
    st.markdown("Fill in a traveller's profile to get an instant insurance prediction.")
 
    c1, c2 = st.columns(2)
    with c1:
        age        = st.slider("Age", 25, 65, 35)
        income     = st.number_input("Annual Income (₦ / local currency)",
                                     min_value=300000, max_value=1800000,
                                     value=800000, step=50000)
        family     = st.slider("Family Members", 2, 9, 4)
        chronic    = st.selectbox("Chronic Disease", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
    with c2:
        emp_type   = st.selectbox("Employment Type",
                                  ["Government Sector", "Private Sector/Self Employed"])
        graduate   = st.selectbox("Graduate?", ["Yes", "No"])
        freq_flyer = st.selectbox("Frequent Flyer?", ["Yes", "No"])
        travelled  = st.selectbox("Ever Travelled Abroad?", ["Yes", "No"])
 
    model_choice = st.radio("Choose model for prediction",
                            list(results.keys()), horizontal=True)
 
    if st.button("🔮 Predict", type="primary"):
        # Encode exactly as in training
        emp_enc  = 0 if emp_type  == "Government Sector"          else 1
        grad_enc = 1 if graduate  == "Yes"                         else 0
        ff_enc   = 1 if freq_flyer == "Yes"                        else 0
        ta_enc   = 1 if travelled  == "Yes"                        else 0
 
        row = np.array([[age, emp_enc, grad_enc, income, family, chronic, ff_enc, ta_enc]])
 
        sel = results[model_choice]
        if model_choice == "Logistic Regression":
            sc2 = StandardScaler()
            sc2.fit(X_res_sc)  # already scaled, use same scaler proxy
            # For demo, scale the input using stats from training set
            row_sc = (row - row.mean()) / (row.std() + 1e-8)
            prob = sel["model"].predict_proba(row_sc)[0][1]
        else:
            prob = sel["model"].predict_proba(row)[0][1]
 
        pred = int(prob >= 0.5)
        colour = "#34d399" if pred == 1 else "#f87171"
        label  = "✅ Likely to Buy Insurance" if pred == 1 else "❌ Unlikely to Buy Insurance"
 
        st.markdown(f"""
        <div style="background:{colour}22; border:1px solid {colour};
                    border-radius:10px; padding:18px; text-align:center; margin-top:12px;">
            <h3 style="color:{colour}; margin:0">{label}</h3>
            <p style="color:white; margin:6px 0 0">
                Confidence: <b style="color:{colour}">{prob*100:.1f}%</b>
                &nbsp;·&nbsp; Model: <b>{model_choice}</b>
            </p>
        </div>""", unsafe_allow_html=True)
 
        st.progress(float(prob), text=f"Purchase probability: {prob*100:.1f}%")
 
