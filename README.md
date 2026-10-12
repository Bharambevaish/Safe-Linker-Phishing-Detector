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

## 💻 How to Run Locally
1. **Clone the repo:** `git clone https://github.com/Bharambevaish/Safe-Linker-Phishing-Detector.git`
2. **Setup venv (Python 3.12):** `python -m venv .venv` and activate it.
3. **Install:** `pip install -r requirements.txt`
4. **Add the dataset:** place `dataset_phishing.xls` (or `.csv`) next to `train.py`, or run `python train.py --data path/to/file`.
5. **Train Model:** `python train.py` — creates `voting_model.joblib`, `feature_names.joblib`, `feature_importance.joblib`.
6. **(Optional) Live threat intel:** copy `.env.example` to `.env` and set `GOOGLE_SAFE_BROWSING_API_KEY`.
7. **Launch:** `streamlit run app.py`

## ☁️ Deployment (Streamlit Community Cloud)
1. Train locally and **commit the three `.joblib` files** (the dataset itself stays out of git).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick this repo, branch `main`, file `app.py`.
3. In **Advanced settings** choose **Python 3.12**, and paste `GOOGLE_SAFE_BROWSING_API_KEY = "..."` into **Secrets** if you have one.
4. Click **Deploy**.

> The model files must be trained with the exact package versions in `requirements.txt`, otherwise the app cannot load them.

---
*Developed by Vaishnavi Bharambe - Walchand College of Engineering*
