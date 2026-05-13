import pandas as pd

def generate_business_features(df_visits, df_controls):
    """
    Generates features aggregating user behavior in a 28-day window
    prior to a facility inspection.
    """
    results = []

    # Business constants (anonymized)
    CARD_CHANNELS = {'physical_card', 'nfc_card'}
    APP_CHANNELS = {'qr_code', 'mobile_token', 'api_token'}
    
    # Example verification statuses
    VERIFICATION_STATUSES = ['positive', 'denied', 'no_authorization', 'reader_failure']
    
    # Statuses used for identifying "Heavy Users" anomalies
    HU_STATUSES = ['denied', 'no_authorization']

    # Note: Iterating with iterrows is computationally expensive.
    # For production Big Data environments, this should be refactored 
    # using vectorized operations (e.g., rolling windows or range joins).
    for idx, row in df_controls.iterrows():
        id_obiekt = row['ID obiektu']
        data_kontroli = row['Data_kontroli']
        poczatek_okresu = row['poczatek_okresu']
        
        # Filtering the time window for a given facility
        df_filtr = df_visits[
            (df_visits['id_obiekt'] == id_obiekt) &
            (df_visits['dzien_wizyty'] >= poczatek_okresu) &
            (df_visits['dzien_wizyty'] < data_kontroli)
        ].copy()
        
        # 1. Basic metrics for positive visits
        mask_pozi = df_filtr['typ_weryfikacji'].isin(['pozytywna', 'pozytywne_wejscie_niepotw'])
        df_pozi = df_filtr[mask_pozi].copy()
        
        liczba_wizyt = df_pozi.shape[0]
        liczba_unikalnych_id_karta = df_filtr['id_karta'].nunique()
        liczba_uzytkownikow_pozi = df_pozi['id_karta'].nunique()
        
        # 2. Identification of Heavy Users (HU) > 14 visits
        df_osoby = df_pozi.groupby('id_karta').size()
        osoby_14plus = df_osoby[df_osoby > 14]
        liczba_osob_14plus = len(osoby_14plus)
        suma_wizyt_osob_14plus = osoby_14plus.sum()
        hu_ids_pozi = osoby_14plus.index
        
        # 3. Status occurrence counting
        liczba_chip = df_pozi[df_pozi['typ_online'] == 'Chip'].shape[0]
        
        df_wart = df_filtr[df_filtr['typ_weryfikacji'].isin(VERIFICATION_STATUSES)]
        counts_dict = df_wart['typ_weryfikacji'].value_counts().to_dict()
        statystyka_wartosci = {s: counts_dict.get(s, 0) for s in VERIFICATION_STATUSES}
        
        suma_statusow = sum(statystyka_wartosci.get(s, 0) for s in HU_STATUSES)
        
        # 4. Time distribution of visits
        df_pozi['godzina'] = df_pozi['dzien_wizyty'].dt.hour
        wizyty_0_8 = ((df_pozi['godzina'] >= 0) & (df_pozi['godzina'] <= 8)).sum()
        wizyty_9_16 = ((df_pozi['godzina'] >= 9) & (df_pozi['godzina'] <= 16)).sum()
        wizyty_17_23 = ((df_pozi['godzina'] >= 17) & (df_pozi['godzina'] <= 23)).sum()

        # 5. Access channel changes (typ_online)
        df_sorted = df_filtr.sort_values(['id_karta', 'dzien_wizyty'])
        zmiany_uzytkownikow = (
            df_sorted.groupby('id_karta')['typ_online']
            .apply(lambda x: (x != x.shift()).sum() - 1)
        )
        uzytkownicy_1_zmiana = (zmiany_uzytkownikow == 1).sum()
        uzytkownicy_wiecej_niz_1_zmiana = (zmiany_uzytkownikow > 1).sum()

        # 6. Channel transitions: Application -> Physical Card
        df_seq = df_pozi.sort_values(['id_karta', 'dzien_wizyty']).copy()
        df_seq['prev_typ_online'] = df_seq.groupby('id_karta')['typ_online'].shift()
        
        df_seq['is_app_to_card'] = (
            df_seq['prev_typ_online'].isin(APP_CHANNELS) &
            df_seq['typ_online'].isin(CARD_CHANNELS) &
            (df_seq['prev_typ_online'] != df_seq['typ_online'])
        )
        przejscia_per_user = df_seq.groupby('id_karta')['is_app_to_card'].sum().astype(int)
        
        sum_app2card = int(przejscia_per_user.sum())
        users_1_app2card = (przejscia_per_user == 1).sum()
        users_gt1_app2card = (przejscia_per_user > 1).sum()

        # 7. Status characteristics for Heavy Users
        df_hu_all = df_filtr[df_filtr['id_karta'].isin(hu_ids_pozi)]
        counts_hu_all = df_hu_all['typ_weryfikacji'].value_counts().to_dict()
        
        hu_statusy = {f"hu_{s}": counts_hu_all.get(s, 0) for s in HU_STATUSES}
        hu_suma_statusow = sum(hu_statusy.values())

        # 8. Filtering for purely physical channels
        tylko_chip = 0
        hu_tylko_chip = 0
        if not df_pozi.empty:
            per_user_channels = df_pozi.groupby('id_karta')['typ_online'].apply(lambda s: set(s.dropna()))
            tylko_chip = int(per_user_channels.apply(lambda s: len(s) > 0 and s.issubset(CARD_CHANNELS)).sum())
            
            per_user_channels_hu = per_user_channels[per_user_channels.index.isin(hu_ids_pozi)]
            hu_tylko_chip = int(per_user_channels_hu.apply(lambda s: len(s) > 0 and s.issubset(CARD_CHANNELS)).sum())

        # 9. Business metric aggregation (with division by zero protection)
        safe_div = lambda a, b: float(a) / b if b > 0 else 0.0
        wszystkie_wizyty = df_filtr.shape[0]
        liczba_wizyt_app_pozi = int(df_pozi['typ_online'].isin(APP_CHANNELS).sum())

        wynik = {
            'ID obiektu': id_obiekt,
            'Data_kontroli': data_kontroli,
            'poczatek_okresu': poczatek_okresu,
            'liczba_wizyt': liczba_wizyt,
            'liczba_unikalnych_id_karta': liczba_unikalnych_id_karta,
            'Liczba HU': liczba_osob_14plus,
            'suma_wizyt_HU': suma_wizyt_osob_14plus,
            'liczba_pozytywnych_chip': liczba_chip,
            'wizyty_0_8': wizyty_0_8,
            'wizyty_9_16': wizyty_9_16,
            'wizyty_17_23': wizyty_17_23,
            'uzytkownicy_1_zmiana_typ_online': uzytkownicy_1_zmiana,
            'uzytkownicy_wiecej_niz_1_zmiana_typ_online': uzytkownicy_wiecej_niz_1_zmiana,
            'suma_statusow': suma_statusow,
            'hu_suma_statusow': hu_suma_statusow,
            'suma_przejsc_aplikacja_na_karte': sum_app2card,
            'uzytkownicy_1_przejscie_aplikacja_na_karte': users_1_app2card,
            'uzytkownicy_wiecej_niz_1_przejscie_aplikacja_na_karte': users_gt1_app2card,
            
            # Indicators
            'udzial_wizyt_HU': safe_div(suma_wizyt_osob_14plus, max(1, liczba_wizyt)),
            'udzial_HU_wszystkich': safe_div(liczba_osob_14plus, max(1, liczba_unikalnych_id_karta)),
            'udzial_statusow_HU': safe_div(hu_suma_statusow, max(1, suma_statusow)),
            'udzial_wizyt_0_8': safe_div(wizyty_0_8, liczba_wizyt),
            'udzial_wizyt_9_16': safe_div(wizyty_9_16, liczba_wizyt),
            'udzial_wizyt_17_23': safe_div(wizyty_17_23, liczba_wizyt),
            'tylko_chip': tylko_chip,
            'hu_tylko_chip': hu_tylko_chip,
            'wsp_tylko_chip': safe_div(tylko_chip, liczba_unikalnych_id_karta),
            'wsp_hu_tylko_chip': safe_div(hu_tylko_chip, liczba_osob_14plus),
            'wizyty_na_uzytkownika': safe_div(liczba_wizyt, liczba_uzytkownikow_pozi),
            'liczba_wizyt_app_pozytywne': liczba_wizyt_app_pozi,
            'wsp_wizyt_app_wszystkie': safe_div(liczba_wizyt_app_pozi, wszystkie_wizyty)
        }

        # Extending dictionary with dynamic status columns
        wynik.update(hu_statusy)
        wynik.update(statystyka_wartosci)
        
        # Protecting against missing 'typ_online' counts
        counts_typ_online = df_pozi['typ_online'].value_counts().to_dict()
        wynik.update(counts_typ_online)

        results.append(wynik)

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Preprocessing module ready.")