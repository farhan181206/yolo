import cv2
from ultralytics import YOLO
import time

# Memuat model YOLOv5
model = YOLO('yolov5s.pt')  # Ganti dengan model yang sesuai

# URL CCTV
cctv_url = "rtsp://admin:BengkelIT@192.168.1.64:554/Streaming/Channels/1"
cap = cv2.VideoCapture(cctv_url)

if not cap.isOpened():
    print("Error: Tidak dapat membuka streaming CCTV.")
else:
    frame_count = 0
    process_every_n_frames = 5  # Proses setiap N frame (misal setiap 5 frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame.")
            break

        frame_count += 1

        # Hanya proses frame setiap N frame
        if frame_count % process_every_n_frames == 0:
            # Mulai waktu untuk menghitung kecepatan deteksi
            start_time = time.time()

            # Melakukan deteksi
            results = model(frame)

            # Menampilkan hasil deteksi
            annotated_frame = results[0].plot()

            # Menampilkan waktu deteksi
            detection_time = time.time() - start_time
            cv2.putText(annotated_frame, f'Detection Time: {detection_time:.2f}s', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Menampilkan frame (dari deteksi atau asli)
        cv2.imshow('Deteksi CCTV', annotated_frame if frame_count % process_every_n_frames == 0 else frame)

        # Menunggu input untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
