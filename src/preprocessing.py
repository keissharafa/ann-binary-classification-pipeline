import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(filepath='data/raw/BlaBla.csv', output_path='data/processed/cleaned_data.csv'):
    print("==========================================================")
    print("       STEP 1: DATA PREPROCESSING & ENCODING UMUR         ")
    print("==========================================================\n")
    
    # Load Dataset
    try:
        df = pd.read_csv(filepath, sep=',')
    except Exception:
        df = pd.read_csv(filepath)

    # Membersihkan spasi ekstra pada seluruh nama kolom
    df.columns = df.columns.str.strip()
    print(f"[INFO] Kolom yang terdeteksi di dataset: {df.columns.tolist()}")

    # Drop kolom 'A' agar fitur input hanya Umur (UMUR_TAHUN) dan B sampai M
    if 'A' in df.columns:
        df = df.drop(columns=['A'])
        print("[INFO] Kolom 'A' dihapus (fitur input: UMUR_TAHUN, B s.d. M).")

    # Custom Encoding Kolom UMUR_TAHUN (Kategori 1 s.d. 5)
    if 'UMUR_TAHUN' in df.columns:
        print("[INFO] Melakukan custom encoding pada kolom 'UMUR_TAHUN'...")
        # Konversi paksa ke numerik (teks non-angka/spasi akan jadi NaN lalu diisi median)
        df['UMUR_TAHUN'] = pd.to_numeric(df['UMUR_TAHUN'], errors='coerce')
        df['UMUR_TAHUN'] = df['UMUR_TAHUN'].fillna(df['UMUR_TAHUN'].median())

        conditions = [
            (df['UMUR_TAHUN'] <= 20),
            (df['UMUR_TAHUN'] >= 21) & (df['UMUR_TAHUN'] <= 30),
            (df['UMUR_TAHUN'] >= 31) & (df['UMUR_TAHUN'] <= 40),
            (df['UMUR_TAHUN'] >= 41) & (df['UMUR_TAHUN'] <= 50),
            (df['UMUR_TAHUN'] > 50)
        ]
        choices = [1, 2, 3, 4, 5]
        df['UMUR_TAHUN'] = np.select(conditions, choices, default=1)
    else:
        print("[WARN] Kolom 'UMUR_TAHUN' tidak ditemukan dalam dataset.")

    # Handling Missing Values
    print("[INFO] Mengecek dan menangani Missing Values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Handling Outlier (IQR Capping)
    print("[INFO] Menangani Outliers pada kolom numerik (Metode IQR Capping)...")
    exclude_from_capping = ['UMUR_TAHUN', 'N', 'target'] 
    for col in numeric_cols:
        if col not in exclude_from_capping:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Encoding Sisa Kolom Kategorikal
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    # Ekspor Data Bersih
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessing selesai. Data disimpan di: '{output_path}'\n")
    
    return df