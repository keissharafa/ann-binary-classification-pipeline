import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def run_splitting_and_training(df, target_column='N'):
    print("==========================================================")
    print("       STEP 2 & 3: SPLITTING, SCALING & ANN TRAINING      ")
    print("==========================================================\n")
    
    # Pemisahan Fitur (X) dan Label (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Data Training: {X_train.shape[0]} baris, Data Testing: {X_test.shape[0]} baris")
    
    # Feature Scaling menggunakan StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Membangun Model Sequential ANN
    input_dim = X_train_scaled.shape[1]
    print("[INFO] Membangun arsitektur Sequential ANN...")
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Output biner
    ])
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Proses Training (100 Epochs)
    print("\n[INFO] Memulai proses training model (100 Epochs)...")
    history = model.fit(
        X_train_scaled, 
        y_train, 
        epochs=100, 
        validation_split=0.2, 
        verbose=1
    )
    
    # Simpan model
    os.makedirs('assets', exist_ok=True)
    model.save('assets/ann_model.keras')
    print("[INFO] Model berhasil dilatih dan disimpan ke 'assets/ann_model.keras'\n")
    
    return model, X_test_scaled, y_test