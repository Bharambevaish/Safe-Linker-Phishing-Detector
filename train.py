import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Import the advanced feature extraction logic (shared with the Streamlit app)
from features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CANDIDATES = ["dataset_phishing.xls", "dataset_phishing.csv", "dataset_phishing.xlsx"]


def find_dataset(csv_path=None):
    if csv_path:
        return csv_path
    for name in DEFAULT_CANDIDATES:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "🚨 Dataset not found. Put dataset_phishing.xls (or .csv) next to train.py, "
        "or pass a path: python train.py --data path/to/file.csv"
    )


def read_dataset(path):
    """
    The Kaggle 'dataset_phishing.xls' is actually a CSV file with an .xls name,
    so try CSV first and only fall back to a real Excel reader.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", dtype=str)
        if "url" in [c.lower() for c in df.columns]:
            return df
    except Exception:
        pass
    return pd.read_excel(path, dtype=str)


def prepare_data(csv_path=None, sample_size=10000):
    """
    Loads data and extracts features exactly the way the Streamlit app does.
    """
    path = find_dataset(csv_path)
    print(f"Loading data from {path}...")
    df = read_dataset(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle original notebook labelling
    if "status" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"status": "label"})
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset needs 'url' and 'status'/'label' columns. Found: {list(df.columns)[:10]}")

    df["label"] = df["label"].astype(str).str.strip().str.lower().replace({
        "bad": "malicious", "phishing": "malicious", "phish": "malicious", "1": "malicious",
        "legitimate": "benign", "good": "benign", "safe": "benign", "0": "benign",
    })
    df = df[df["label"].isin(["benign", "malicious"])].dropna(subset=["url"])
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both benign and malicious URLs.")

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    print(f"Extracting features from {len(df)} URLs (this may take a moment)...")
    extracted = df["url"].apply(extract_features).tolist()

    # Drop URLs that could not be parsed instead of training on all-zero rows
    keep = [bool(f) for f in extracted]
    dropped = len(keep) - sum(keep)
    if dropped:
        print(f"Skipping {dropped} unparseable URLs.")
    features_df = pd.DataFrame([f for f in extracted if f]).fillna(0)
    labels = df["label"][keep]

    # Target variable (1 for malicious, 0 for benign)
    y = (labels == "malicious").astype(int).values
    return features_df.values, y, features_df.columns.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Safe Linker phishing model")
    parser.add_argument("--data", help="Path to the dataset (default: dataset_phishing.xls next to this file)")
    parser.add_argument("--sample", type=int, default=10000, help="Max rows to train on (0 = all)")
    args = parser.parse_args()

    print("=" * 50)
    print("SAFE LINKER - Major Engineering Project Model Training")
    print("=" * 50)

    X, y, feature_names = prepare_data(args.data, args.sample)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Voting Classifier (Random Forest + XGBoost, soft voting)...")
    # Soft voting outputs probabilities instead of hard predictions, key for the 0-100 hybrid score.
    # VotingClassifier fits its own copies of the base models, so they are not trained separately.
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  eval_metric="logloss", random_state=42)),
        ],
        voting="soft",
        weights=[1, 1.5],  # Slightly more weight to XGBoost
    )
    voting_clf.fit(X_train, y_train)

    # Evaluation
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Voting Classifier Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "malicious"]))

    # Save the model and feature names so Streamlit can ensure column order matches
    output_model = os.path.join(BASE_DIR, "voting_model.joblib")
    output_features = os.path.join(BASE_DIR, "feature_names.joblib")
    output_importance = os.path.join(BASE_DIR, "feature_importance.joblib")

    joblib.dump(voting_clf, output_model, compress=3)
    joblib.dump(feature_names, output_features)

    # Feature importance from the XGBoost model that is actually inside the ensemble
    xgb_fitted = voting_clf.named_estimators_["xgb"]
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": xgb_fitted.feature_importances_})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    joblib.dump(importance_df, output_importance)

    print(f"\nModels saved:\n  {output_model}\n  {output_features}\n  {output_importance}")
    print("You can now run 'streamlit run app.py' to start the dashboard!")
