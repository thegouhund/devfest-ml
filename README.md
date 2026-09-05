# 🏥 Family Health Monitor: AI Anomaly Detection Engine

Repositori ini berisi *pipeline Machine Learning* untuk sistem peringatan dini kesehatan keluarga. Sistem ini bertugas mendeteksi anomali pada tanda-tanda vital (*vital signs*) seperti detak jantung dan laju pernapasan, dengan mempertimbangkan konteks personal aktivitas pengguna.

Proyek ini dirancang sebagai sistem pakar di *Backend* yang memproses data dari kamera rPPG (*Frontend*) dan ekstraksi konteks dari Chatbot LLM.

---

## 🧠 Arsitektur Machine Learning (The "Why")

Mendeteksi anomali kesehatan tidak bisa menggunakan aturan baku (*Rule-Based*) seperti "BPM > 100 = Bahaya". BPM 130 sangat wajar bagi seseorang yang baru selesai berlari, namun sangat mematikan bagi lansia yang sedang tidur.

Oleh karena itu, kami membangun sistem berbasis **Unsupervised Machine Learning** dengan pendekatan berikut:

1. **Isolation Forest:** Algoritma *Unsupervised* ringan yang dirancang khusus untuk memisahkan/mengisolasi titik data yang aneh (*outliers*) dari kerumunan kebiasaan normal pengguna. Sangat hemat komputasi untuk *real-time Backend processing*.
2. **Medical Feature Engineering:** Alih-alih menyuapi model dengan data mentah, kami meracik fitur cerdas:
   - `delta_bpm`: Selisih antara BPM saat ini dengan rata-rata *baseline* normal pengguna.
   - `delta_rr`: Selisih laju pernapasan.
   - `bpm_to_rr_ratio`: Rasio kardiovaskular.
3. **Ordinal Encoding (Anti-Curse of Dimensionality):** Chatbot LLM kami bertugas menyimpulkan cerita aktivitas pengguna menjadi **Tingkat Aktivitas** (Skala 0 - 3). Angka inilah yang masuk ke model ML. Cara ini mencegah redundansi data (*Feature Bloat*) yang sering menghancurkan akurasi algoritma *Unsupervised*.

---

## 📊 Metodologi Evaluasi (Method 1: Synthetic Outlier Injection)

Di dunia nyata, tidak ada label "Sakit" atau "Sehat" di data vital mentah (*Unsupervised*). Namun, untuk membuktikan kehandalan model di ajang Hackathon ini, kami memvalidasi model secara *offline* menggunakan metodologi standar emas riset medis:

*   Kami membuat 10.000 data sintetis dengan menyuntikkan 4% anomali medis buatan (*Tachycardia, Bradycardia*).
*   Model Isolation Forest dilatih secara murni **Buta (Tanpa Label)**.
*   Kunci Jawaban rahasia kemudian dibuka HANYA di tahap evaluasi untuk menghitung metrik **PR-AUC** dan mencari batas *threshold* menggunakan **F2-Score** (Metrik yang mengutamakan *Recall* / penyelamatan nyawa di atas segalanya).
*   **Hasil Akhir:** Model mampu mencapai Akurasi 97% dengan PR-AUC 0.87, serta menekan angka *False Alarm* hingga hanya 1.8%.

---

## 📂 Struktur Direktori

```text
devfest-ml/
│
├── data/
│   └── health_dummy_data.csv       # Dataset sintetis dengan injeksi anomali untuk simulasi
│
├── models/
│   └── health_anomaly_model.pkl    # Bundle Model Final (Model, Scaler, Fitur & Threshold)
│
├── notebooks/
│   └── health_anomaly_detection.ipynb # Dokumentasi riset, training, dan visualisasi
│
├── requirements.txt                # Dependensi Python
├── .gitignore                      
└── README.md                       
```

---

## 🚀 Cara Menjalankan Proyek (Setup Lokal)

Jika Anda ingin mereplikasi penelitian atau menjalankan *Jupyter Notebook* secara lokal:

1. **Clone repositori ini**
2. **Buat Virtual Environment:**
   ```bash
   python -m venv .venv
   ```
3. **Aktivasi Environment:**
   *   Windows: `.\.venv\Scripts\activate`
   *   Mac/Linux: `source .venv/bin/activate`
4. **Install Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
5. Buka `health_anomaly_detection.ipynb` menggunakan Jupyter Notebook atau VS Code.

---

## 🔌 Panduan Integrasi Backend (BE)

Bagi tim Backend (FastAPI / Express), mengintegrasikan ML ini sangatlah mudah. Seluruh "otak" ML telah dibungkus menjadi **1 file berukuran kecil** (`health_anomaly_model.pkl`). Tidak perlu melakukan *hardcode threshold* apapun.

**Contoh Implementasi di Python (FastAPI):**

```python
import joblib
import pandas as pd

# 1. LOAD MODEL (Hanya dilakukan 1x saat Server Startup)
# Bundle ini berisi scaler, model, dan optimal_threshold
model_bundle = joblib.load('models/health_anomaly_model.pkl')
scaler = model_bundle['scaler']
model = model_bundle['model']
threshold = model_bundle['optimal_threshold']

# 2. TERIMA DATA DARI FRONTEND (Contoh Payload)
# Data ini harus sudah melalui perhitungan delta dari database (BPM saat ini - Baseline BPM)
payload = {
    'delta_bpm': [45.0],           # Melonjak 45 angka dari biasanya
    'delta_rr': [4.0],             # Nafas lebih cepat
    'bpm_to_rr_ratio': [6.0],
    'bpm_variance': [20.0],
    'activity_level_score': [0]    # 0 = Sedang Istirahat (Konteks Chatbot)
}
df_input = pd.DataFrame(payload)

# 3. PREDIKSI (Real-time)
# Normalisasi data
data_scaled = scaler.transform(df_input)

# Dapatkan skor anomali mentah
anomaly_score = model.decision_function(data_scaled)[0]

# 4. KEPUTUSAN BISNIS
if -anomaly_score > threshold:
    print("🚨 ANOMALI TERDETEKSI! Kirim Notifikasi Telegram ke Keluarga.")
else:
    print("✅ Vitals Aman. Simpan ke database historis.")
```
