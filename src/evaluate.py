import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test_scaled, y_test):
    print("==========================================================")
    print("         STEP 4: EVALUASI & CONFUSION MATRIX              ")
    print("==========================================================\n")
    
    # Evaluasi Loss & Akurasi
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"[INFO] Akurasi di data uji: {acc * 100:.2f}%")
    print(f"[INFO] Loss di data uji   : {loss:.4f}\n")

    # Prediksi Kelas (Threshold 0.5)
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Laporan Klasifikasi
    print(" CLASSIFICATION REPORT ".center(55, '='))
    print(classification_report(y_test, y_pred))
    print("=" * 55 + "\n")

    # Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Deep Learning (ANN)')
    plt.tight_layout()

    os.makedirs('assets', exist_ok=True)
    plot_path = 'assets/confusion_matrix_dl.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Plot Confusion Matrix disimpan di: '{plot_path}'\n")
    print("==========================================================")
    print("          PIPELINE DEEP LEARNING SELESAI!                 ")
    print("==========================================================\n")