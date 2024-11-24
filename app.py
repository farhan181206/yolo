from flask import Flask, render_template, Response, jsonify
import cv2
import easyocr
from ultralytics import YOLO

# Inisialisasi Flask
app = Flask(__name__)

# Inisialisasi model YOLOv5
model = YOLO('yolov5su.pt')  # Ganti dengan model YOLO yang sesuai

# Inisialisasi EasyOCR untuk membaca plat nomor
reader = easyocr.Reader(['en'])

# URL stream CCTV
stream_url = "parking.mp4"  # Ganti dengan URL CCTV
cap = cv2.VideoCapture(stream_url)

# Area parkir (contoh bounding box area parkir)
parking_area = (100, 100, 400, 400)  # (x1, y1, x2, y2)

# Data slot parkir (contoh)
parking_data = {
    "total": 10,
    "occupied": 3,
    "empty": 7
}

# Fungsi untuk memeriksa apakah kendaraan di area parkir
def is_in_parking_area(vehicle_bbox, parking_area):
    x1, y1, x2, y2 = vehicle_bbox
    px1, py1, px2, py2 = parking_area
    return px1 <= x1 <= px2 and py1 <= y1 <= py2

# Fungsi untuk mendeteksi kendaraan dan membaca plat nomor
def process_frame(frame):
    results = model.predict(frame)  # Deteksi kendaraan dengan YOLO
    annotated_frame = results[0].plot()  # Gambar hasil deteksi

    detected_info = []

    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        label = model.names[int(cls)]  # Nama kelas (mobil/motor)

        # Periksa apakah di area parkir
        parking_status = "Benar" if is_in_parking_area(bbox, parking_area) else "Salah"

        # Potong gambar untuk OCR
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        ocr_results = reader.readtext(cropped_img)

        plate_number = None
        if ocr_results:
            plate_number = ocr_results[0][1]  # Ambil teks plat nomor

        # Simpan informasi deteksi
        detected_info.append({
            "label": label,
            "bbox": bbox,
            "status": parking_status,
            "plate": plate_number
        })

        # Tambahkan bounding box ke frame
        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} | {parking_status}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if plate_number:
            cv2.putText(annotated_frame, f"Plate: {plate_number}", (bbox[0], bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return annotated_frame, detected_info

# Fungsi untuk membaca stream video dan memproses frame
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        processed_frame, detected_info = process_frame(frame)

        # Encode frame untuk streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route utama untuk halaman web
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk streaming video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route untuk informasi slot parkir
@app.route('/slot_info')
def slot_info():
    return jsonify(parking_data)

# Jalankan server Flask
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
















































# import cv2
# from flask import Flask, render_template, Response
# from ultralytics import YOLO
# import threading
# import queue
# import time
 
# app = Flask(__name__)

# # URL RTSP untuk CCTV
# cctv_url = "rtsp://admin:BengkelIT@192.168.1.64:554/Streaming/Channels/1"
# cap = cv2.VideoCapture(cctv_url)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Atur buffer untuk mempercepat pemrosesan
# cap.set(cv2.CAP_PROP_FPS, 30)  # Atur frame rate sesuai dengan kamera CCTV Anda

# # Inisialisasi model YOLO
# model = YOLO("yolov5su.pt")

# # Queue untuk buffer frame
# frame_queue = queue.Queue(maxsize=1)

# # Pengaturan ukuran frame rendah untuk performa maksimal
# frame_width, frame_height = 640, 480
# slot_width = frame_width // 4

# # Slot parkir dengan batasan posisi di frame
# parking_slots = [(i * slot_width, 0, (i + 1) * slot_width, frame_height) for i in range(4)]

# # Batasi frekuensi deteksi
# DETECTION_INTERVAL = 0.5  # Setengah detik
# last_detection_time = time.time()

# def capture_frames():
#     while True: 
#         success, frame = cap.read()
#         if not success:
#             break
#         frame = cv2.resize(frame, (frame_width, frame_height))
#         if not frame_queue.full():
#             frame_queue.put(frame)

# def generate_frames():
#     global last_detection_time
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()

#             # Lakukan deteksi hanya setiap DETECTION_INTERVAL detik
#             current_time = time.time()
#             if current_time - last_detection_time >= DETECTION_INTERVAL:
#                 last_detection_time = current_time
#                 results = model(frame)

#                 occupied_count = 0
#                 occupied_slots = []

#                 # Periksa setiap slot parkir
#                 for idx, (x1, y1, x2, y2) in enumerate(parking_slots):
#                     slot_occupied = False
#                     for result in results:
#                         for box in result.boxes.xyxy:
#                             bx1, by1, bx2, by2 = box[:4]
#                             if (x1 < bx1 < x2) and (y1 < by1 < y2):
#                                 slot_occupied = True
#                                 occupied_count += 1
#                                 occupied_slots.append(idx + 1)
#                                 break

#                     # Gambar rectangle untuk slot parkir
#                     color = (0, 0, 255) if slot_occupied else (0, 255, 0)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#                 # Tambahkan teks untuk status parkir
#                 cv2.putText(frame, f'Parking spaces: {len(parking_slots)}/{occupied_count}',
#                             (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#                 cv2.putText(frame, f'Kosong: Slot {", ".join(str(i) for i in range(1, len(parking_slots)+1) if i not in occupied_slots)}',
#                             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#                 cv2.putText(frame, f'Terisi: Slot {", ".join(map(str, occupied_slots))}',
#                             (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#             # Encode frame sebagai JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     threading.Thread(target=capture_frames, daemon=True).start()
#     app.run(debug=True, host='0.0.0.0', port=5000)
