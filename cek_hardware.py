import torch
from ultralytics import YOLO

# Cek apakah GPU tersedia
if torch.cuda.is_available():
    print("Program berjalan di GPU:", torch.cuda.get_device_name(0))
else:
    print("Program berjalan di CPU")

# Inisialisasi model YOLO dan pindahkan ke GPU jika tersedia
model = YOLO("yolov5s.pt")

# Pindahkan model ke GPU jika ada
if torch.cuda.is_available():
    model = model.to('cuda')
    print("Model berhasil dipindahkan ke GPU")
else:
    print("Model tetap di CPU")
