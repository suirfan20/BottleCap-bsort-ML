from ultralytics import YOLO



model = YOLO("artifacts/bsort_train/weights/best.pt")  # ganti path sesuai punyamu


print("nc:", model.model.nc)
print("names:", model.names)


results = model("data/val/images/raw-250110_dc_s001_b4_3.jpg", conf=0.006, save=True)
print(results[0].boxes)
print(results[0].boxes.conf)