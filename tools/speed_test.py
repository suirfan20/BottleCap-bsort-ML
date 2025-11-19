from ultralytics import YOLO
import cv2
import time

model = YOLO("artifacts/bsort_train/weights/best.pt")

img = cv2.imread("data/val/images/raw-250110_dc_s001_b4_1.jpg")

# Warmup 3x (biar hasil stabil)
for _ in range(3):
    model.predict(img, imgsz=320)

# hitung 100x inference
N = 100
start = time.time()

for _ in range(N):
    model.predict(img, imgsz=320, verbose=False)

end = time.time()

ms_per_frame = (end - start) / N * 1000
fps = 1000 / ms_per_frame

print(f"Inference speed: {ms_per_frame:.2f} ms/frame")
print(f"FPS: {fps:.2f}")
