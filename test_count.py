import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO('yolov5s.pt')

# Callback untuk mendapatkan koordinat RGB dari mouse
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load video
cap = cv2.VideoCapture('parking1.mp4')

# Load daftar kelas COCO
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Area parkir (definisi polygon untuk setiap area)
areas = {
    1: [(52, 364), (30, 417), (73, 412), (88, 369)],
    2: [(105, 353), (86, 428), (137, 427), (146, 358)],
    3: [(159, 354), (150, 427), (204, 425), (203, 353)],
    4: [(217, 352), (219, 422), (273, 418), (261, 347)],
    5: [(274, 345), (286, 417), (338, 415), (321, 345)],
    6: [(336, 343), (357, 410), (409, 408), (382, 340)],
    7: [(396, 338), (426, 404), (479, 399), (439, 334)],
    8: [(458, 333), (494, 397), (543, 390), (495, 330)],
    9: [(511, 327), (557, 388), (603, 383), (549, 324)],
    10: [(564, 323), (615, 381), (654, 372), (596, 315)],
    11: [(616, 316), (666, 369), (703, 363), (642, 312)],
    12: [(674, 311), (730, 360), (764, 355), (707, 308)]
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Predict using YOLO
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # List untuk jumlah mobil di tiap area dan mobil salah parkir
    counts = {i: 0 for i in areas.keys()}
    wrong_parked = 0

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])
        class_name = class_list[class_id]

        if 'car' in class_name:
            # Pusat bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_area = False  # Flag untuk cek apakah mobil berada di area

            for area_id, points in areas.items():
                result = cv2.pointPolygonTest(np.array(points, np.int32), (cx, cy), False)
                if result >= 0:
                    # Tambahkan kotak hijau dan titik merah untuk deteksi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    counts[area_id] += 1
                    cv2.putText(frame, str(class_name), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    in_area = True
                    break

            # Jika tidak di area mana pun, hitung sebagai salah parkir
            if not in_area:
                wrong_parked += 1
                # Tambahkan kotak merah untuk mobil salah parkir
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Wrong Parking", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Total area kosong
    total_areas = len(areas)
    occupied = sum(counts.values())
    free_spaces = total_areas - occupied

    print(f"Total free spaces: {free_spaces}, Wrong parked cars: {wrong_parked}")

    # Visualisasi setiap area
    for area_id, points in areas.items():
        color = (0, 255, 0) if counts[area_id] == 0 else (0, 0, 255)
        cv2.polylines(frame, [np.array(points, np.int32)], True, color, 2)
        cv2.putText(frame, str(area_id), points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Tampilkan jumlah mobil salah parkir di frame
    cv2.putText(frame, f"Wrong parked: {wrong_parked}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow("RGB", frame)

    # Tambahkan delay untuk memperlambat video
    time.sleep(0.03)  # Tambah atau kurangi untuk mengatur kecepatan

    # Break jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resource
cap.release()
cv2.destroyAllWindows()
