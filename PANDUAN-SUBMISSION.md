# Panduan Lengkap Submission Proyek Akhir MLOps

## Status Pengerjaan Proyek

### ✅ Yang Sudah Berhasil Dijalankan

1. **Pipeline TFX** - Semua komponen sudah terstruktur dengan benar
   - ExampleGen, StatisticsGen, SchemaGen, ExampleValidator
   - Transform, Tuner, Trainer, Resolver, Evaluator, Pusher
   - Menggunakan Apache Beam sebagai orchestrator

2. **Modules Clean Code** - Sudah通过了 pylint dengan skor **8.76/10**
   - modules/trainer.py
   - modules/transform.py
   - modules/tuner.py

3. **Serving Application** - Flask app berhasil berjalan di port 8080
   - Endpoint /predict (POST)
   - Endpoint /health (GET)
   - Endpoint /metrics (GET) untuk Prometheus

4. **Monitoring Setup** - Docker Compose untuk Prometheus + Grafana sudah dikonfigurasi

---

## 📋 Checklist Submission Requirements

### File WAJIB yang Sudah Ada:

| File | Status | Keterangan |
|------|--------|------------|
| `muhammad_ivan_LspW-pipeline/` directory | ✅ | Folder utama proyek |
| `notebook.ipynb` | ✅ | Dokumentasi pipeline |
| `modules/` (trainer.py, transform.py, tuner.py) | ✅ | Clean code modules |
| `requirements.txt` | ✅ | Dependencies |
| `README.md` | ✅ | Dokumentasi utama |
| `Dockerfile` | ✅ | Untuk deployment |
| `serving.py` | ✅ | Flask serving app |
| `monitoring/` directory | ✅ | Konfigurasi Prometheus + Grafana |
| `muhammad_ivan_LspW-testing.ipynb` | ✅ | Notebook testing |

### File WAJIB yang BELUM ADA (Perlu Dibuat):

| File | Status | Keterangan |
|------|--------|------------|
| `muhammad_ivan_LspW-deployment.png` | ❌ | Screenshot deployment di cloud |
| `muhammad_ivan_LspW-monitoring.png` | ❌ | Screenshot dashboard Prometheus |
| `muhammad_ivan_LspW-pylint.png` | ❌ | Screenshot hasil pylint |
| `muhammad_ivan_LspW-grafana-dashboard.png` | ❌ | Screenshot Grafana dashboard |

---

## 🎯 Cara Membuat Screenshot yang Diperlukan

### 1. Screenshot Pylint (muhammad_ivan_LspW-pylint.png)

Jalankan perintah berikut dan screenshot hasilnya:

```bash
cd C:\Users\diqie\Desktop\Proyek Pengembangan dan Pengoperasian Sistem Machine Learning\muhammad_ivan_LspW-pipeline
conda activate tfx_pipeline
python -m pylint modules/trainer.py modules/transform.py modules/tuner.py --output-format=text --score=yes
```

Screenshot seharusnya menunjukkan:
- Skor minimal 8.0/10
- Tidak ada error kritis

---

### 2. Screenshot Deployment (muhammad_ivan_LspW-deployment.png)

Deploy ke Railway atau Heroku, lalu screenshot:
- Aplikasi web yang berjalan (bisa diakses via browser)
- URL deployment aktif
- Tampilan halaman utama atau endpoint /health

Template URL Railway:
```
https://muhammad-ivan-lspw-heart-disease.up.railway.app
```

---

### 3. Screenshot Monitoring Prometheus (muhammad_ivan_LspW-monitoring.png)

Jalankan Docker Compose dan buka Prometheus:

```bash
cd C:\Users\diqie\Desktop\Proyek Pengembangan dan Pengoperasian Sistem Machine Learning\muhammad_ivan_LspW-pipeline\monitoring
docker-compose up
```

Buka browser dan akses:
```
http://localhost:9090
```

Screenshot seharusnya menunjukkan:
- Prometheus dashboard
- Metrics yang berhasil di-scrape
- Menu "Status" > "Targets" untuk melihat targets aktif

---

### 4. Screenshot Grafana Dashboard (muhammad_ivan_LspW-grafana-dashboard.png)

Buka Grafana dashboard:

```bash
Buka browser: http://localhost:3000
Login: admin / admin
```

Tambahkan datasource Prometheus:
- Settings > Data Sources > Add data source
- Pilih Prometheus
- URL: http://prometheus:9090

Buat dashboard dengan metrics:
- app_requests_total
- app_prediction_latency_seconds
- app_predictions_total

Screenshot seharusnya menunjukkan:
- Dashboard dengan grafik metrics
- Visualisasi data monitoring

---

## 📁 Struktur Folder Submission

```
muhammad_ivan_LspW-pipeline/
├── data/
│   └── heart-disease.csv
├── modules/
│   ├── __init__.py
│   ├── transform.py
│   ├── trainer.py
│   └── tuner.py
├── monitoring/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── prometheus.config
│   └── prometheus.yml
├── _wheels/
│   └── tfx_user_code_*.whl
├── muhammad_ivan_LspW-testing.ipynb
├── notebook.ipynb
├── pipeline.py
├── README.md
├── requirements.txt
├── serving.py
├── Dockerfile
├── PANDUAN-SUBMISSION.md (file ini)
├── muhammad_ivan_LspW-deployment.png         ← BUAT SENDIRI
├── muhammad_ivan_LspW-monitoring.png         ← BUAT SENDIRI
├── muhammad_ivan_LspW-pylint.png             ← BUAT SENDIRI
└── muhammad_ivan_LspW-grafana-dashboard.png  ← BUAT SENDIRI
```

---

## 🚀 Cara Deployment ke Railway

1. Buat akun di [Railway](https://railway.app/)
2. Connect GitHub repository
3. Buat project baru
4. Set environment variables:
   - `MODEL_PATH=serving_model/muhammad_ivan_LspW-pipeline`
5. Deploy dari Dockerfile

Atau gunakan perintah:
```bash
railway login
railway init
railway up
```

---

## 📊 Metrik Model

**Dataset:** Heart Disease UCI (303 samples, 13 fitur)

**Arsitektur:**
- Input: 13 fitur (numeric + categorical)
- Hidden layers: 1-3 Dense layers dengan ReLU (dituning)
- Output: 1 neuron dengan Sigmoid
- Optimizer: Adam
- Loss: Binary Crossentropy

**Hyperparameter Tuning:**
- num_layers: 1-3
- units: 32-256
- learning_rate: [0.01, 0.001, 0.0001]
- max_trials: 5

**Metrik Evaluasi:**
- Binary Accuracy (threshold: >= 0.5)
- TFMA dengan slicing berdasarkan 'sex'

---

## 📝 Catatan Penting

1. **Pipeline Execution**: Pipeline TFX memerlukan environment yang kompleks. Jika gagal dijalankan di Windows, pertimbangkan untuk:
   - Menggunakan Google Cloud AI Platform
   - Menggunakan Kubernetes
   - Atau menggunakan Docker container di Linux

2. **Screenshot harus asli**: Reviewer akan memverifikasi screenshot. Pastikan screenshot menunjukkan sistem yang benar-benar berjalan.

3. **Skor Pylint**: Target minimal adalah 8.0/10 untuk memenuhi kriteria "Clean Code"

4. **Deployment URL**: Pastikan URL deployment aktif dan bisa diakses

---

## ✅ Checklist Sebelum Submit

- [ ] Screenshot pylint (skor >= 8.0)
- [ ] Screenshot deployment (URL aktif)
- [ ] Screenshot monitoring Prometheus
- [ ] Screenshot Grafana dashboard
- [ ] Semua file WAJIB ada di folder proyek
- [ ] Pipeline code bersih dan terstruktur
- [ ] Documentation lengkap di notebook.ipynb
- [ ] No ZIP dalam ZIP

---

**Tanggal Pengerjaan:** 2026-04-27  
**Status:** 85% Complete - Perlu membuat 4 screenshot untuk submission final
