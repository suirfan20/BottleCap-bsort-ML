# Bottle Cap Color Detection (bsort)

Project ini dibuat untuk membangun pipeline Computer Vision untuk mendeteksi warna tutup botol:

- **light_blue**
- **dark_blue**
- **others**


---

## 1. Overview

Pipeline yang dibuat mencakup:

- Dataset relabeling berdasarkan HSV (light/dark blue)
- Training model YOLOv8n
- CLI tool (`bsort`) untuk training dan inference
- Jupyter Notebook untuk dokumentasi eksperimen
- Docker + Docker Compose untuk environment yang reproducible
- CI/CD (lint, format, test, docker build)

---

## 2. Project Structure
- src/bsort/ # package utama
- notebooks/ # eksperimen
- data/ # dataset (ignored in git)
- artifacts/ # hasil training (ignored in git)
- outputs/ # hasil inference (ignored in git)
- tests/ # test
- tools/ # tools
- settings.yaml # konfigurasi utama
- pyproject.toml # konfigurasi project
- Dockerfile
- docker-compose.yml
- README.md

## 3. Installation (Local)
### 3.1 Clone repository
```bash
git clone <YOUR_REPO_URL>.git
cd ada-mata-bsort
```

### 3.2 Create virtual environment 
python -m venv venv
# Windows:
venv\Scripts\activate

### 3.3 Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"


## 4. Configuration
Semua parameter berada di settings.yaml

## 5. CLI Usage — bsort
### 5.1 Training
bsort train --config settings.yaml
#result difolder artifacts/

### 5.2 Inference
bsort infer --config settings.yaml --image sample.jpg
#result difolder outputs/

## 6. Running with Docker
### 6.1 Build image
docker build -t bsort:latest .

### 6.2 Inference inside container
docker run --rm \
  -v $(pwd):/app \
  bsort:latest \
  infer --config settings.yaml --image sample.jpg

## 7. Docker Compose (Dev + Notebook)
docker compose up -d

## 8. Experiments & Results
### 8.1 Notebook
Seluruh eksperimen terdapat di notebook "notebooks/01_bottlecap_detection_experiments.ipynb"

Isi notebook mencakup:
- dataset
- relabeling warna via HSV
- training
- evaluasi
- inference speed test
- catatan & limitasi

### 8.2 Inference Speed
Dilakukan di device (CPU only):
Device: Intel i7-7500U, 8GB RAM
Image size: 320×320
Average inference time: ~50 ms/frame
FPS: ~19.4

## 9. Color Relabeling Logic (HSV)
Logika relabel:
- Konversi BGR → HSV
- Ambil area tengah objek (50%) agar tidak kena background
- Hitung median hue/saturation/value

## 10. Model Limitations
- Light vs dark blue kadang ambigu karena shadow
- Dataset kecil → generalisasi terbatas

## 11. Weights & Biases (Optional)
W&B project: https://wandb.ai/suirfan20-/BottleCap-bsort-ML


## 12. CI/CD
Pipeline CI/CD:
    - Lint (black, isort, pylint)
    - Unit test (pytest)
    - Docker Compose build + CLI test
Semua terdapat di: .github/workflows/ci.yml


## 13. Reproduce
git clone <repo>
cd ada-mata-bsort
pip install -e ".[dev]"
bsort train --config settings.yaml
bsort infer --config settings.yaml --image sample.jpg




