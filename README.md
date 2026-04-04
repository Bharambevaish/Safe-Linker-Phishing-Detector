# Safe Linker: Hybrid Phishing URL Detection System
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%2B%20RF-green.svg)](https://xgboost.readthedocs.io/)

**Safe Linker** is a high-performance cybersecurity tool designed to detect malicious and phishing URLs in real-time. By utilizing a **Soft-Voting Ensemble** of XGBoost and Random Forest, the system analyzes 15+ URL features to provide a weighted risk assessment.

## 🚀 Key Features
- **Hybrid Detection:** Combines Rule-based heuristics with advanced Machine Learning.
- **Ensemble Architecture:** Uses a Voting Classifier (XGBoost + Random Forest) for 95%+ accuracy.
- **Deep Feature Engineering:** Extracts entropy, brand mimicry, and structural URL patterns.
- **Real-time Dashboard:** Built with Streamlit for instant security reporting and visualization.
- **Security Deep Dive:** Provides a "Reasoning" accordion explaining *why* a link was flagged.

## 🛠️ Tech Stack
- **Languages:** Python
- **ML Libraries:** Scikit-Learn, XGBoost, Joblib
- **Data Handling:** Pandas, NumPy
- **Frontend:** Streamlit, Plotly (for Feature Importance charts)

## 📁 Project Structure
- `app.py`: The Streamlit dashboard and UI logic.
- `train.py`: The ML pipeline (Data loading -> Preprocessing -> Training).
- `features.py`: The core engine for advanced feature extraction.
- `requirements.txt`: List of dependencies for environment reproducibility.

## 💻 How to Run
1. **Clone the repo:** `git clone https://github.com/Bharambevaish/Safe-Linker.git`
2. **Setup Venv:** `python -m venv .venv` and activate it.
3. **Install:** `pip install -r requirements.txt`
4. **Train Model:** `python train.py`
5. **Launch:** `streamlit run app.py`

---
*Developed by Vaishnavi Bharambe - Walchand College of Engineering*
