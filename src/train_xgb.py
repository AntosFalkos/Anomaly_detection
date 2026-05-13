import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ---------- CONFIGURATION ----------
SEED = 42
DATA_PATH = "../data/synthetic_anomalies_data.csv"
IMAGES_DIR = "../images"
ENC = "utf-8-sig"

# XGBoost Tuning params
SMOTE_SAMPLING   = 0.65   # Weaker SMOTE to reduce FP
SMOTE_K          = 5
CLASS0_WEIGHT    = 1.15   # Slightly penalize minority class to boost TN
MIN_RECALL_1     = 0.90   # Minimum acceptable recall
THR_GRID         = np.linspace(0.05, 0.80, 2501)

np.random.seed(SEED)

def pick_threshold_max_acc(y_true, proba, min_recall1=0.90, grid=THR_GRID, tie_eps=1e-6):
    """Selects threshold maximizing Accuracy given a minimum recall constraint."""
    best_acc, best_t = -1.0, 0.5
    for t in grid:
        yhat = (proba >= t).astype(int)
        rec1 = recall_score(y_true, yhat, zero_division=0)
        if rec1 < min_recall1:
            continue
        acc = accuracy_score(y_true, yhat)
        if (acc - best_acc) > 1e-12 or (abs(acc - best_acc) <= tie_eps and t > best_t):
            best_acc, best_t = acc, t
    if best_acc < 0:
        accs = [(accuracy_score(y_true, (proba>=t).astype(int)), t) for t in grid]
        best_t = max(accs)[1]
    return float(best_t)

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH, encoding=ENC)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}. Please ensure synthetic data is present.")
        return

    columns_to_drop = ['Data_kontroli', 'poczatek_okresu', 'ID obiektu']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    df['Fraud'] = pd.to_numeric(df['Fraud'], errors='coerce').fillna(0).astype(int)
    X = df.drop(columns=['Fraud'])
    y = df['Fraud']

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )

    print(f"TRAIN: {X_train.shape[0]} samples, TEST: {X_test.shape[0]} samples")

    # 3. SMOTE & Model Training
    print("Applying SMOTE and training XGBoost model...")
    sm = SMOTE(random_state=SEED, k_neighbors=SMOTE_K, sampling_strategy=SMOTE_SAMPLING)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)

    weights = np.where(y_res == 0, CLASS0_WEIGHT, 1.0)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        min_child_weight=13,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=12.0,
        gamma=2.0,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_res, y_res, sample_weight=weights)

    # 4. Threshold Optimization
    proba_va = model.predict_proba(X_va)[:, 1]
    thr = pick_threshold_max_acc(y_va, proba_va, min_recall1=MIN_RECALL_1, grid=THR_GRID)
    print(f"Optimal Decision Threshold (Validation): {thr:.3f}")

    # 5. Evaluation
    proba_te = model.predict_proba(X_test)[:, 1]
    y_pred_te = (proba_te >= thr).astype(int)

    print("\n=== FINAL TEST RESULTS (XGBoost) ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_te):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred_te, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(y_test, y_pred_te, zero_division=0):.2%}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_te, zero_division=0):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_te, digits=4))

    # 6. Plotting & Saving
    print("Generating plots...")
    
    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_te, display_labels=[0, 1], cmap="Blues")
    plt.title(f"Confusion Matrix (Threshold={thr:.3f})")
    plt.savefig(os.path.join(IMAGES_DIR, "xgb_confusion_matrix.png"))
    plt.close()

    # Feature Importance (Top 15)
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 7))
    plt.barh([features[i] for i in indices][::-1], importances[indices][::-1])
    plt.xlabel("Feature Importance Score")
    plt.title("Top 15 Feature Importances (XGBoost)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "xgb_feature_importance.png"))
    plt.close()
    
    print(f"Process complete. Artifacts saved in '{IMAGES_DIR}/' directory.")

if __name__ == "__main__":
    main()