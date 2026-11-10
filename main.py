import sys
import os

# Agar folder 'src' terbaca oleh sistem
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import run_preprocessing
from src.model import run_splitting_and_training
from src.evaluate import evaluate_model

if __name__ == '__main__':
    print("[INFO] Memulai Orkestrasi Pipeline Deep Learning (ANN)...")
    
    # 1. Preprocessing Data & Custom Encoding Umur
    df_cleaned = run_preprocessing(
        filepath='data/raw/BlaBla.csv', 
        output_path='data/processed/cleaned_data.csv'
    )
    
    # 2 & 3. Splitting, Feature Scaling, & Model Training (100 Epochs)
    model, X_test_s, y_test = run_splitting_and_training(df_cleaned, target_column='N')
    
    # 4. Evaluasi Model & Plot Confusion Matrix
    evaluate_model(model, X_test_s, y_test)