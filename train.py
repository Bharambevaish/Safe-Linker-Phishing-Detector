import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import the advanced feature extraction logic
from features import extract_features

def prepare_data(csv_path=None):
    """
    Loads data and extracts advanced features so they exactly match the Streamlit app.
    Falls back to a smaller synthetic run if the dataset is not found in the same folder.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "dataset_phishing.xls")

    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"🚨 Missing {csv_path} in the specified directory. Please ensure the dataset is present before running.")
        
    try:
        df = pd.read_excel(csv_path)
    except Exception:
        # Fallback in case the .xls file is actually a comma-separated CSV text file internally
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip', dtype=str)
    # Handle original notebook labelling
    if 'status' in df.columns:
        df.rename(columns={'status': 'label'}, inplace=True)
        
    df['label'] = df['label'].astype(str).str.lower()
    df['label'] = df['label'].replace({
        'bad': 'malicious', 'phishing': 'malicious', 'phish': 'malicious',
        'legitimate': 'benign', 'good': 'benign', 'safe': 'benign'
    })
    df = df[df['label'].isin(['benign', 'malicious'])]
    df = df.sample(min(10000, len(df)), random_state=42) # sample for speed in initial engineering run

    print("Extracting advanced features (this may take a moment)...")
    # Apply exactly the same feature extraction as features.py
    extracted_features = df['url'].apply(extract_features)
    
    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(extracted_features.tolist())
    
    # Ensure all missing values are 0
    features_df = features_df.fillna(0)
    
    # Target variable (1 for malicious, 0 for benign)
    y = df['label'].apply(lambda x: 1 if x == 'malicious' else 0).values
    X = features_df.values
    
    return X, y, features_df.columns.tolist()

if __name__ == "__main__":
    print("="*50)
    print("SAFE LINKER - Major Engineering Project Model Training")
    print("="*50)
    
    # You can update this path to the correct dataset path if needed.
    X, y, feature_names = prepare_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Base Models...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    
    # Train
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    print("\nTraining Voting Classifier (Hybrid Score Optimization)...")
    # Soft Voting outputs probabilities instead of hard predictions, key for our Hybrid 0-100 Score
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft',
        weights=[1, 1.5] # Give slightly more weight to XGBoost
    )
    
    voting_clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Voting Classifier Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and feature names so Streamlit can ensure column order matches
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_model = os.path.join(base_dir, "voting_model.joblib")
    output_features = os.path.join(base_dir, "feature_names.joblib")
    output_importance = os.path.join(base_dir, "feature_importance.joblib")
    
    joblib.dump(voting_clf, output_model)
    joblib.dump(feature_names, output_features)
    
    # Extract feature importance from the XGB base estimator
    xgb_importance = xgb_model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": xgb_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    joblib.dump(importance_df, output_importance)
    
    print(f"Models successfully saved to {output_model}, {output_features}, and {output_importance}.")
    print("You can now run 'streamlit run app.py' to start the dashboard!")
