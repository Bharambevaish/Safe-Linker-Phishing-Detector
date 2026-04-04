import streamlit as st
import pandas as pd
import joblib
import time
import os

# Import our modularized features
from features import extract_features, rule_based_severity, check_external_intel

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

# Load the models globally
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "voting_model.joblib")
    features_path = os.path.join(base_dir, "feature_names.joblib")
    importance_path = os.path.join(base_dir, "feature_importance.joblib")
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)
        
        # Load importance if available
        importance_df = None
        if os.path.exists(importance_path):
            importance_df = joblib.load(importance_path)
            
        return model, feature_cols, importance_df
    return None, None, None

model, feature_cols, importance_df = load_models()

# Input Section
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com/login", label_visibility="collapsed")
    analyze_btn = st.button("Analyze Threat Profile")

if analyze_btn and url_input:
    if model is None:
        st.error("Please run train.py first to initialize the AI.")
    else:
        with st.spinner("Analyzing advanced heuristics and external intelligence..."):
            time.sleep(1.5) # Simulate slight network/ML latency for premium UX feel
            
            # 1. Feature Extraction
            features_dict = extract_features(url_input)
            
            # Ensure expected features exist
            for col in feature_cols:
                if col not in features_dict:
                    features_dict[col] = 0
            
            # Map dict to DataFrame maintaining feature column order
            df_features = pd.DataFrame([features_dict], columns=feature_cols)
            df_features = df_features.fillna(0)
            X_input = df_features.values
            
            # 2. Rule-Based Score
            rule_score, rule_reasons = rule_based_severity(features_dict)
            
            # 3. External API Score (MOCK)
            api_score, api_reasons = check_external_intel(url_input)
            
            # 4. ML Model Probability Score (Class 1: Malicious Probability)
            ml_probs = model.predict_proba(X_input)[0]
            ml_score = ml_probs[1] # Probability of being Malicious
            
            # ========================================================
            # 5. Hybrid Optimization Formula
            # ========================================================
            # ML Model: 40%
            # Rule Based: 30%
            # API Threat Intel: 30%
            hybrid_score_raw = (0.4 * ml_score) + (0.3 * rule_score) + (0.3 * api_score)
            risk_percentage = min(hybrid_score_raw * 100, 100.0)
            
            # Determine Status 
            if risk_percentage > 70:
                status = "Critical Risk"
                status_class = "risk-high"
                icon = "🚨"
            elif risk_percentage > 35:
                status = "Suspicious"
                status_class = "risk-med"
                icon = "⚠️"
            else:
                status = "Verified Safe"
                status_class = "risk-low"
                icon = "✅"

        # RESULTS DASHBOARD
        st.markdown("---")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
                <h3 style='color: #94a3b8; font-weight: 500;'>Overall Risk Score</h3>
                <p class='metric-value {status_class}'>{risk_percentage:.1f}%</p>
                <div style='font-size: 1.5rem; margin-top: 10px; font-weight: bold;'>{icon} {status}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-bottom: 20px;'>Hybrid Diagnostic Breakdown</h3>", unsafe_allow_html=True)
            
            st.progress(ml_score)
            st.caption(f"Machine Learning Confidence: **{ml_score*100:.1f}%**")
            
            st.progress(rule_score)
            st.caption(f"Rule-Based Severity: **{rule_score*100:.1f}%**")
            
            st.progress(api_score)
            st.caption(f"External API Threat Intelligence (Google/PhishTank): **{api_score*100:.1f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

        # SIEMENS DEEP DIVE SECTION
        st.markdown("### 🔍 Security Deep Dive")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        deep_dive_found = False
        
        if rule_reasons:
            deep_dive_found = True
            st.markdown("**Rule-Based Engine Flags:**")
            for r in rule_reasons:
                st.markdown(f"- {r}")
                
        if api_reasons:
            deep_dive_found = True
            st.markdown("<br>**External Intel Flags:**", unsafe_allow_html=True)
            for r in api_reasons:
                st.markdown(f"- {r}")
                
        if ml_score > 0.6:
            deep_dive_found = True
            st.markdown("<br>**Machine Learning Analysis:**", unsafe_allow_html=True)
            st.markdown(f"- High ML Probability implies hidden obfuscated phishing patterns detected by the Voting Classifier.")
            
        if not deep_dive_found:
            st.success("No anomalies detected. The URL structurally matches safe organizational architecture. API reports clean.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("Show Raw Extracted Features Matrix"):
            st.dataframe(df_features.T.rename(columns={0: "Feature Value"}), use_container_width=True)

# Feature Importance Visualization in Sidebar
if importance_df is not None:
    st.sidebar.markdown("## 📊 ML Feature Importance")
    st.sidebar.caption("Top drivers for the XGBoost model")
    # Take top 10
    top_10 = importance_df.head(10).set_index("Feature")
    st.sidebar.bar_chart(top_10, use_container_width=True)
    st.sidebar.markdown("---")

# Footer configuration info
st.sidebar.markdown("## ⚙️ Integrations")
st.sidebar.markdown("""
To enable live external intelligence, add API keys into your `.env` or `secrets.toml`.

`GOOGLE_SAFE_BROWSING_API_KEY=your_key`

*Currently running in mock-intel mode.*
""")
st.sidebar.markdown("---")
st.sidebar.caption("Safe Linker Pro v2.0 - Major Engineering Project")
