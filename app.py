import streamlit as st
import pandas as pd
import joblib
import os

# Import our modularized features
from features import extract_features, rule_based_severity, check_external_intel, normalize_url

# Configure page settings
st.set_page_config(
    page_title="Safe Linker | Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium SaaS aesthetics and Glassmorphism
st.markdown("""
<style>
    :root {
        --primary-bg: #0f172a;
        --secondary-bg: #1e293b;
        --accent: #3b82f6;
        --text-light: #f8fafc;
        --danger: #ef4444;
        --warning: #f59e0b;
        --success: #10b981;
    }
    
    .stApp {
        background-color: var(--primary-bg);
        color: var(--text-light);
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        padding: 0;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }

    .risk-high { color: var(--danger); text-shadow: 0 0 10px rgba(239,68,68,0.5); }
    .risk-med { color: var(--warning); text-shadow: 0 0 10px rgba(245,158,11,0.5); }
    .risk-low { color: var(--success); text-shadow: 0 0 10px rgba(16,185,129,0.5); }

    /* Custom Input Styling */
    div[data-baseweb="input"] > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Analyze Button */
    div.stButton > button {
        background-color: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# Application Header
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🛡️ Safe Linker Enterprise</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 3rem;'>AI-Powered Advanced URL Phishing Threat Intelligence</p>", unsafe_allow_html=True)

def get_secret(name):
    """Read a key from environment / .env, or from a Streamlit secrets.toml if one exists."""
    if os.environ.get(name):
        return os.environ[name]
    # Only touch st.secrets when a secrets file exists; otherwise Streamlit
    # prints a red "No secrets found" error on the page.
    secrets_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml"),
        os.path.expanduser(os.path.join("~", ".streamlit", "secrets.toml")),
    ]
    if any(os.path.exists(p) for p in secrets_paths):
        try:
            return st.secrets.get(name)
        except Exception:
            return None
    return None


def load_dotenv_file():
    """Minimal .env loader so local runs pick up keys without extra packages."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv_file()
GOOGLE_API_KEY = get_secret("GOOGLE_SAFE_BROWSING_API_KEY")


# Load the models globally
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "voting_model.joblib")
    features_path = os.path.join(base_dir, "feature_names.joblib")
    importance_path = os.path.join(base_dir, "feature_importance.joblib")

    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        return None, None, None, "Model files not found."

    try:
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)
        importance_df = joblib.load(importance_path) if os.path.exists(importance_path) else None
    except Exception as e:
        # Usually a scikit-learn/xgboost version mismatch between training and deployment
        return None, None, None, f"Model files could not be loaded ({type(e).__name__}: {e})."

    return model, feature_cols, importance_df, None


model, feature_cols, importance_df, model_error = load_models()

if model is None:
    st.error(
        f"{model_error} Run `python train.py` with the same package versions as "
        "`requirements.txt`, then commit voting_model.joblib, feature_names.joblib "
        "and feature_importance.joblib."
    )
    st.stop()

# Input Section
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    with st.form("analyze_form", border=False):
        url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com/login", label_visibility="collapsed")
        analyze_btn = st.form_submit_button("Analyze Threat Profile")

url_input = (url_input or "").strip()

if analyze_btn and not url_input:
    st.warning("Please enter a URL first.")

if analyze_btn and url_input:
    with st.spinner("Analyzing advanced heuristics and external intelligence..."):
        # 1. Feature Extraction
        features_dict = extract_features(url_input)

    if not features_dict:
        st.error("That doesn't look like a valid URL. Try something like `https://example.com/login`.")
        st.stop()

    with st.spinner("Scoring..."):
        # Build the model input in the exact column order used during training
        row = {col: features_dict.get(col, 0) for col in feature_cols}
        df_features = pd.DataFrame([row], columns=feature_cols).fillna(0)
        X_input = df_features.values

        # 2. Rule-Based Score
        rule_score, rule_reasons = rule_based_severity(features_dict)

        # 3. External Threat Intel (live Google Safe Browsing if a key is set)
        api_score, api_reasons, intel_mode = check_external_intel(url_input, google_api_key=GOOGLE_API_KEY)

        # 4. ML Model Probability Score (Class 1: Malicious Probability)
        ml_score = float(model.predict_proba(X_input)[0][1])

        # ========================================================
        # 5. Hybrid Optimization Formula
        # ========================================================
        # ML Model: 40% | Rule Based: 30% | Threat Intel: 30%
        hybrid_score_raw = (0.4 * ml_score) + (0.3 * rule_score) + (0.3 * api_score)
        risk_percentage = min(max(hybrid_score_raw * 100, 0.0), 100.0)

        if risk_percentage > 70:
            status, status_class, icon = "Critical Risk", "risk-high", "🚨"
        elif risk_percentage > 35:
            status, status_class, icon = "Suspicious", "risk-med", "⚠️"
        else:
            status, status_class, icon = "Likely Safe", "risk-low", "✅"

    # RESULTS DASHBOARD
    st.markdown("---")
    st.caption(f"Analyzed: `{normalize_url(url_input)}`")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <h3 style='color: #94a3b8; font-weight: 500;'>Overall Risk Score</h3>
            <p class='metric-value {status_class}'>{risk_percentage:.1f}%</p>
            <div style='font-size: 1.5rem; margin-top: 10px; font-weight: bold;'>{icon} {status}</div>
        </div>
        """, unsafe_allow_html=True)

    intel_label = "Google Safe Browsing (live)" if intel_mode == "live" else "Threat Intel (offline heuristic)"

    with col_res2:
        with st.container(border=True):
            st.markdown("### Hybrid Diagnostic Breakdown")

            st.progress(min(max(ml_score, 0.0), 1.0))
            st.caption(f"Machine Learning Confidence: **{ml_score*100:.1f}%**")

            st.progress(min(max(float(rule_score), 0.0), 1.0))
            st.caption(f"Rule-Based Severity: **{rule_score*100:.1f}%**")

            st.progress(min(max(float(api_score), 0.0), 1.0))
            st.caption(f"{intel_label}: **{api_score*100:.1f}%**")

    # SECURITY DEEP DIVE SECTION
    st.markdown("### 🔍 Security Deep Dive")
    with st.container(border=True):
        deep_dive_found = False

        if rule_reasons:
            deep_dive_found = True
            st.markdown("**Rule-Based Engine Flags:**")
            for r in rule_reasons:
                st.markdown(f"- {r}")

        if api_reasons:
            deep_dive_found = True
            st.markdown("**External Intel Flags:**")
            for r in api_reasons:
                st.markdown(f"- {r}")

        if ml_score > 0.6:
            deep_dive_found = True
            st.markdown("**Machine Learning Analysis:**")
            st.markdown("- High ML probability: the Voting Classifier matched this URL's structure to phishing patterns it learned during training.")

        if not deep_dive_found:
            st.success("No anomalies detected. The URL's structure looks normal and no threat-intel flags were raised.")

    with st.expander("Show Raw Extracted Features Matrix"):
        st.dataframe(df_features.T.rename(columns={0: "Feature Value"}), use_container_width=True)

# Feature Importance Visualization in Sidebar
if importance_df is not None:
    st.sidebar.markdown("## 📊 ML Feature Importance")
    st.sidebar.caption("Top drivers for the XGBoost model")
    top_10 = importance_df.head(10).set_index("Feature")
    st.sidebar.bar_chart(top_10, use_container_width=True)
    st.sidebar.markdown("---")

# Integrations status
st.sidebar.markdown("## ⚙️ Integrations")
if GOOGLE_API_KEY:
    st.sidebar.success("Google Safe Browsing: API key configured (falls back to the offline heuristic if a lookup fails)")
else:
    st.sidebar.info(
        "Google Safe Browsing: not configured — using the offline heuristic.\n\n"
        "Add `GOOGLE_SAFE_BROWSING_API_KEY` to `.env` (local) or the app's Secrets (Streamlit Cloud) to enable live checks."
    )
st.sidebar.markdown("---")
st.sidebar.caption("Safe Linker Pro v2.1 - Major Engineering Project")
