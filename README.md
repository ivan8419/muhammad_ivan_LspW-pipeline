# Submission 2: Prediksi Penyakit Jantung dengan MLOps Pipeline

Nama: Muhammad Ivan

Username dicoding: muhammad_ivan_LspW

| | Deskripsi |
| ----------- | ----------- |
| **Dataset** | [Heart Disease Dataset (UCI)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) — Dataset klinis pasien berisi 303 sampel dengan 13 fitur untuk memprediksi keberadaan penyakit jantung. |
| **Masalah** | Penyakit jantung merupakan penyebab kematian utama di dunia. Deteksi dini sangat penting untuk meningkatkan angka keselamatan pasien. Proyek ini bertujuan membangun sistem prediksi otomatis yang dapat mendeteksi risiko penyakit jantung berdasarkan data klinis pasien seperti usia, tekanan darah, kolesterol, dan kondisi EKG. |
| **Solusi machine learning** | Membangun pipeline machine learning end-to-end menggunakan TensorFlow Extended (TFX) dengan 9 komponen utama: ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Resolver (Evaluator baseline), Evaluator, dan Pusher. Pipeline dijalankan menggunakan Apache Beam sebagai orchestrator. |
| **Metode pengolahan** | Fitur numerik (age, trestbps, chol, thalach, oldpeak) dinormalisasi ke rentang [0, 1] menggunakan `tft.scale_to_0_1`. Fitur kategorikal (sex, cp, fbs, restecg, exang, slope, ca, thal) digunakan langsung tanpa encoding tambahan. Label target di-cast ke int64. |
| **Arsitektur model** | Neural Network dengan 13 input features. Hidden layers: 1–3 Dense layer dengan aktivasi ReLU (jumlah layer dan unit per layer dituning secara otomatis menggunakan Keras Tuner RandomSearch dengan 5 trials). Output layer: 1 neuron dengan aktivasi Sigmoid untuk klasifikasi biner. Optimizer: Adam dengan learning rate yang dituning dari {0.01, 0.001, 0.0001}. Loss: Binary Crossentropy. |
| **Metrik evaluasi** | Binary Accuracy (threshold minimal 0.5) dievaluasi menggunakan TensorFlow Model Analysis (TFMA). Evaluasi dilakukan secara keseluruhan dan per slice berdasarkan fitur `sex`. Model hanya akan di-push ke serving jika melewati threshold. |
| **Performa model** | Model dievaluasi menggunakan TFMA dengan metrik Binary Accuracy. Berdasarkan hasil tuning hyperparameter dengan Keras Tuner (5 trials), model terbaik mencapai akurasi validasi pada dataset Heart Disease. Hasil evaluasi TFMA menunjukkan model lulus threshold (Binary Accuracy >= 0.5). |
| **Opsi deployment** | Model dideploy menggunakan **Flask** sebagai web server yang dikemas dalam **Docker container**. Platform cloud yang digunakan adalah **Railway** (alternatif Heroku). Model disimpan dalam format TensorFlow SavedModel dan diakses melalui endpoint REST API `/predict`. |
| **Web app** | `https://muhammad-ivan-lspw-heart-disease.railway.app` — Endpoint prediksi: `/predict` (POST), Health check: `/health` (GET), Metrics: `/metrics` (GET). |
| **Monitoring** | Sistem dimonitor menggunakan **Prometheus** yang dikonfigurasi untuk scraping metrics dari aplikasi Flask di port 8001. Metrics yang dipantau: `app_requests_total` (total request per endpoint dan status), `app_prediction_latency_seconds` (latensi prediksi), dan `app_predictions_total` (total prediksi per label). Dashboard monitoring divisualisasikan menggunakan **Grafana** yang terhubung ke Prometheus datasource, dapat diakses di port 3000. |

---

## Struktur Direktori

```
muhammad_ivan_LspW-pipeline/
├── data/
│   └── heart-disease.csv           # Dataset Heart Disease
├── modules/
│   ├── __init__.py
│   ├── transform.py                # Modul preprocessing TFX
│   ├── trainer.py                  # Modul training model
│   └── tuner.py                    # Modul hyperparameter tuning
├── monitoring/
│   ├── Dockerfile                  # Dockerfile untuk Prometheus
│   ├── docker-compose.yml          # Orchestrasi Prometheus + Grafana + ML App
│   ├── prometheus.yml              # Konfigurasi scraping Prometheus
│   └── prometheus.config           # Konfigurasi Prometheus
├── serving_model/                  # Model yang telah di-push oleh pipeline
├── notebook.ipynb                  # Notebook dokumentasi pipeline
├── muhammad_ivan_LspW-testing.ipynb  # Notebook pengujian prediction request
├── pipeline.py                     # Script TFX pipeline utama
├── serving.py                      # Flask web server untuk model serving
├── Dockerfile                      # Dockerfile untuk serving app
├── requirements.txt                # Dependensi Python
└── README.md                       # Dokumentasi ini
```

## Cara Menjalankan Pipeline

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan pipeline TFX
python pipeline.py

# 3. Jalankan serving app secara lokal
python serving.py

# 4. Jalankan monitoring (Prometheus + Grafana)
cd monitoring
docker-compose up -d
```

## Saran yang Diterapkan

1. ✅ **Hyperparameter Tuning** — Komponen `Tuner` dengan Keras Tuner RandomSearch (5 trials, tuning: num_layers, units, learning_rate)
2. ✅ **Clean Code** — Kode dipisahkan ke direktori `modules/` dengan docstrings lengkap, type hints, dan pylint score >= 7.0
3. ✅ **Notebook Pengujian** — `muhammad_ivan_LspW-testing.ipynb` berisi kode untuk melakukan prediction request ke model yang di-deploy di cloud
4. ✅ **Grafana Dashboard** — Grafana disinkronkan dengan Prometheus melalui `docker-compose.yml` untuk dashboard monitoring yang lebih menarik
"# muhammad_ivan_LspW-pipeline" 
