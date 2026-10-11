import os

def run_pipeline():
    print("🚀 Rozpoczynanie pipeline'u detekcji anomalii...\n")
    
    print("--- KROK 1: Preprocessing i Inżynieria Cech ---")
    os.system("python src/preprocess.py")
    
    print("\n--- KROK 2: Trening i ewaluacja modelu XGBoost ---")
    os.system("python src/train_xgb.py")
    
    print("\n--- KROK 3: Trening i ewaluacja modelu MLP ---")
    os.system("python src/train_mlp.py")
    
    print("\n✅ Pipeline zakończony pomyślnie.")

if __name__ == "__main__":
    run_pipeline()