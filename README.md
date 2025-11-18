# bsort — Bottle Cap Color Detection

📌 **GitHub Repository**  
https://github.com/suirfan20/BottleCap-bsort-ML

📌 **Docker Image (optional)**  
ghcr.io/suirfan20/bsort:latest

## Overview
This project is a real-time bottle cap detector…


# bsort – Bottle Cap Color Detection

Project ini mendeteksi 3 warna tutup botol:

- `light_blue`
- `dark_blue`
- `others`

Pipeline menggunakan YOLOv8 yang ringan sehingga cocok untuk edge device (misalnya Raspberry Pi 5).

---

## 1. Cara Instalasi

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"


2. Cara Training
bsort train --config settings.yaml


Model terbaik akan disimpan di:

artifacts/best.pt

3. Cara Inference
bsort infer --config settings.yaml --image path/to/image.jpg


Hasil disimpan di:

outputs/<nama>_pred.jpg

4. Notebook

Notebook eksperimen ada di:

notebooks/01_experiments_bottlecap.ipynb


Isinya:

Eksplorasi dataset

Relabel warna (HSV)

Visualisasi bounding box

Training model via CLI

Evaluasi model (mAP, Confusion Matrix)

Benchmark inference time

5. Struktur Project
src/bsort/   # kode utama: CLI, model, infer, train
notebooks/   # eksperimen & analisis
artifacts/   # output training
outputs/     # hasil inference
settings.yaml
README.md

6. Inference Time (Isi setelah test)

Contoh:

Model: YOLOv8n
Image Size: 320
Inference Time: 6.2 ms/frame (CPU)

7. Catatan & Improvement

Relabel warna masih heuristic: perlu tuning

Bisa ditingkatkan dengan ONNX/quantization

Bisa ditambah data real dari edge device


---

# 🔥 Sekarang apa yang kamu lakukan?

1. **Bikin notebook** → copy 12 cell tadi (kode + markdown).  
2. **Bikin README** → copy template di atas.  
3. Kalau ada yang bingung dari cell notebook → kasih tau gue.  
4. Kalau mau gue bikinin *file .ipynb jadi langsung bisa download*, bilang aja.  

Gue temenin sampai kelar 100%. 💪
