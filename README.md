# 🏥 Family Health Monitor: AI Anomaly Detection Engine

Repositori ini berisi _pipeline Machine Learning_ untuk sistem peringatan dini kesehatan keluarga. Sistem ini bertugas mendeteksi anomali pada tanda-tanda vital (_vital signs_) seperti detak jantung dan laju pernapasan, dengan mempertimbangkan konteks personal aktivitas pengguna.

Proyek ini dirancang sebagai **Sistem Pakar di Backend** yang memproses data dari kamera rPPG (_Frontend_) dan ekstraksi konteks dari Chatbot LLM.

---

## 🧠 1. Mengapa Machine Learning? (vs. Rule-Based Tradisional)

Standar kesehatan umum (seperti pedoman detak jantung 60–100 BPM) memiliki kelemahan fatal di dunia nyata yang kami sebut sebagai **'The One-Size-Fits-All Flaw'**. Di sinulah _Machine Learning_ (Isolation Forest) kami bekerja mengatasinya:

- **Menghindari False Negatives (Kebobolan):** Jika lansia memiliki _baseline_ detak jantung 55 BPM, dan tiba-tiba naik tajam ke 92 BPM saat sedang beristirahat. Sistem tradisional (Rule-based) akan menganggap 92 BPM sebagai "Normal" dan kebobolan. ML kami menghitung selisih (`delta_bpm`) dan akan seketika membunyikan alarm karena sadar itu adalah lonjakan abnormal untuk lansia tersebut.
- **Menghindari False Positives (Alarm Palsu):** Jika lansia memiliki _baseline_ 85 BPM, dan naik ke 102 BPM karena sedang senang. Sistem tradisional langsung membunyikan sirine bahaya (karena >100). ML kami memaklumi hal ini karena masih masuk dalam rentang variansi amannya.
- **Ruang Multi-Dimensi:** ML kami tidak hanya melihat 1 angka. Algoritma kami menggabungkan _BPM, Laju Pernapasan, Variabilitas (HRV), dan Tingkat Aktivitas (Skala 0-3 dari LLM Chatbot)_ secara bersamaan. Sesuatu yang mustahil dilakukan oleh ribuan baris aturan `IF-ELSE` tradisional.

---

## 🛡️ 2. Arsitektur Pertahanan ML (ML Defenses)

Untuk memastikan keandalan sistem secara jangka panjang, arsitektur ML ini dilengkapi dengan 3 lapisan solusi spesifik:

### A. Extended Healthy Calibration (Menangani Data Kotor di Awal)

**Masalah:** Bagaimana jika lansia sedang sakit parah selama 14 hari pertama penggunaan aplikasi?
**Solusi:** Kami menerapkan _Medical Gatekeeper_. Selama masa kalibrasi 14 hari, data yang masuk disaring oleh batasan medis umum. Jika suatu hari lansia terdeteksi sakit (misal: BPM 120 saat istirahat), hari tersebut **TIDAK DIHITUNG** ke dalam kalibrasi ML. Masa kalibrasi akan otomatis diperpanjang hingga terkumpul "14 Hari Data Sehat Terkonfirmasi".

### B. Continuous Learning / Sliding Window (Menangani Perubahan Fisik)

**Masalah:** Bagaimana jika detak jantung lansia berubah secara permanen di masa depan karena makin rutin berolahraga?
**Solusi:** Model ML ini dieksekusi menggunakan strategi **Sliding 14-Day Window**. Server secara periodik me-_retrain_ (`.fit()`) model secara otomatis hanya menggunakan data 14 hari ke belakang. _Baseline_ akan dinamis bergerak mengikuti kesehatan terbaru lansia (Mengatasi _Concept Drift_).

### C. Explainable AI / SHAP (Mengatasi Algoritma Black-Box)

**Masalah:** Bagaimana cara menjelaskan penyebab alarm kepada orang awam?
**Solusi:** ML kami diintegrasikan dengan pustaka **SHAP (SHapley Additive exPlanations)**. ML secara matematis menghitung bobot setiap organ (Misal: _delta_bpm_ menyumbang 70% penyebab bahaya, _activity_level_ 30%). Angka-angka XAI ini kemudian dikirim ke LLM DeepSeek, yang akan merangkumnya menjadi bahasa manusia: _"Alarm berbunyi karena detak jantung kakek melonjak jauh saat sedang beristirahat."_

---

## 📊 3. Metodologi Evaluasi (Method 1: Synthetic Outlier Injection)

Model **Isolation Forest** ini dilatih murni **Tanpa Label (Unsupervised)**. Namun, untuk membuktikan kehandalannya, kami memvalidasinya secara _offline_ (Standar Emas Riset):

- Kami membangun dataset 10.000 titik data sintetis dan menyuntikkan 4% anomali serangan jantung/napas buatan.
- Kunci Jawaban rahasia dibuka di akhir untuk menghitung metrik performa.
- **Hasil Akhir Uji Klinis Offline:**
  - **Akurasi Keseluruhan:** 97%
  - **PR-AUC Score:** 0.871 (Sangat tinggi untuk deteksi anomali multi-variabel)
  - **Recall:** 82% (Berhasil menyelamatkan mayoritas besar kasus anomali buatan)
  - **False Alarm Rate:** Hanya ~1.8%

---

## 📂 4. Struktur Direktori

```text
devfest-ml/
│
├── data/
│   └── health_dummy_data.csv       # Dataset sintetis dengan injeksi anomali untuk simulasi
│
├── models/
│   └── health_anomaly_model.pkl    # Bundle Final (Isi: Scaler, Model, Fitur, & Threshold 0.0805)
│
├── notebooks/
│   └── health_anomaly_detection.ipynb # Riset, training, PR-AUC, F2-Score, & Plot SHAP (XAI)
│
├── requirements.txt                # Dependensi Python (pandas, scikit-learn, shap)
├── .gitignore
└── README.md
```

---

## 🔌 5. Panduan Integrasi Backend (FastAPI / Express)

Seluruh "otak" ML ini telah dibungkus menjadi **1 file berukuran sangat kecil** (`health_anomaly_model.pkl`). Backend (BE) tidak perlu melakukan _hardcode threshold_ maupun _scaling_ manual.

**Contoh _Snippet_ Implementasi di Python (FastAPI):**

```python
import joblib
import pandas as pd

# 1. LOAD MODEL BUNDLE (Hanya dijalankan 1x saat Server Startup)
model_bundle = joblib.load('models/health_anomaly_model.pkl')
scaler = model_bundle['scaler']
model = model_bundle['model']
threshold = model_bundle['optimal_threshold']

# 2. TERIMA DATA DARI FRONTEND (Contoh Payload saat Scan Selesai)
# Catatan: BE harus menghitung delta berdasarkan database sebelum dimasukkan ke ML
payload = {
    'delta_bpm': [45.0],           # Melonjak 45 angka dari baseline 14-hari
    'delta_rr': [4.0],             # Nafas lebih cepat
    'bpm_to_rr_ratio': [6.0],
    'bpm_variance': [20.0],
    'activity_level_score': [0]    # 0 = Konteks LLM (Sedang Istirahat)
}
df_input = pd.DataFrame(payload)

# 3. PREDIKSI OTOMATIS (Real-time)
data_scaled = scaler.transform(df_input)
anomaly_score = model.decision_function(data_scaled)[0]

# 4. KEPUTUSAN (Alerting)
if -anomaly_score > threshold:
    print("🚨 ANOMALI TERDETEKSI! Kirim Notifikasi Telegram dan Generate XAI.")
else:
    print("✅ Vitals Aman. Simpan ke time-series database.")
```
