import os
import numpy as np
import pandas as pd

def generate_mock_data(num_facilities=2000, num_visits=150000):
    np.random.seed(42)
    
    # 1. Generowanie tabeli kontroli (df_controls)
    facility_ids = [f"FAC_{i:04d}" for i in range(num_facilities)]
    end_date = pd.to_datetime("2026-10-11")
    start_date = end_date - pd.Timedelta(days=28)
    
    df_controls = pd.DataFrame({
        'ID obiektu': facility_ids,
        'Data_kontroli': [end_date] * num_facilities,
        'poczatek_okresu': [start_date] * num_facilities,
        'is_anomaly': np.random.choice([0, 1], num_facilities, p=[0.97, 0.03]) # 3% anomalii
    })
    
    # 2. Generowanie tabeli wizyt (df_visits)
    # Statusy dopasowane do list z preprocess.py
    verification_types = [
        'pozytywna', 'pozytywne_wejscie_niepotw', 
        'positive', 'denied', 'no_authorization', 'reader_failure'
    ]
    
    online_types = [
        'physical_card', 'nfc_card', 'Chip', 
        'qr_code', 'mobile_token', 'api_token'
    ]
    
    # Generowanie losowych momentów w czasie w oknie 28 dni
    random_days = np.random.randint(0, 28, num_visits)
    random_hours = np.random.randint(0, 24, num_visits)
    random_minutes = np.random.randint(0, 60, num_visits)
    
    visit_dates = (start_date + 
                   pd.to_timedelta(random_days, unit='D') + 
                   pd.to_timedelta(random_hours, unit='h') + 
                   pd.to_timedelta(random_minutes, unit='m'))
    
    # Pula użytkowników ograniczona, aby wygenerować "Heavy Users" (>14 wizyt)
    user_ids = [f"USR_{i:04d}" for i in range(num_visits // 30)]
    
    df_visits = pd.DataFrame({
        'id_obiekt': np.random.choice(facility_ids, num_visits),
        'dzien_wizyty': visit_dates,
        'typ_weryfikacji': np.random.choice(
            verification_types, 
            num_visits, 
            p=[0.70, 0.10, 0.05, 0.05, 0.05, 0.05]
        ),
        'id_karta': np.random.choice(user_ids, num_visits),
        'typ_online': np.random.choice(online_types, num_visits)
    })
    
    # 3. Zapis do plików
    os.makedirs("data", exist_ok=True)
    df_controls.to_csv("data/df_controls.csv", index=False)
    df_visits.to_csv("data/df_visits.csv", index=False)
    
    print(f"Wygenerowano {num_facilities} kontroli oraz {num_visits} wizyt w katalogu 'data/'.")

if __name__ == "__main__":
    generate_mock_data()