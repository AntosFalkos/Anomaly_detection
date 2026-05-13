import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ---------- CONFIGURATION ----------
SEED = 42
DATA_PATH = "../data/synthetic_anomalies_data.csv"
IMAGES_DIR = "../images"
ENC = "utf-8-sig"

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    # 3. Scaling (Critical for Neural Networks)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Neural Network Architecture Definition
    print("Building and training the MLP model...")
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # 5. Training
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=30,  # Training limit based on early stabilization
        batch_size=32,
        verbose=1
    )

    # 6. Evaluation
    proba_mlp = model.predict(X_test_scaled).flatten()
    mlp_y_pred = (proba_mlp > 0.5).astype(int)

    print("\n=== FINAL TEST RESULTS (MLP Keras) ===")
    print("Classification Report:")
    print(classification_report(y_test, mlp_y_pred, digits=4))

    # 7. Plotting & Saving
    print("Generating plots...")

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, mlp_y_pred, cmap="Blues")
    plt.title("Confusion Matrix – MLP Classifier")
    plt.grid(False)
    plt.savefig(os.path.join(IMAGES_DIR, "mlp_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba_mlp)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – MLP Classifier')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(IMAGES_DIR, "mlp_roc_curve.png"))
    plt.close()

    # Learning Curves (Loss)
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Log-Loss across Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(IMAGES_DIR, "mlp_learning_curves.png"))
    plt.close()

    print(f"Process complete. Artifacts saved in '{IMAGES_DIR}/' directory.")

if __name__ == "__main__":
    main()