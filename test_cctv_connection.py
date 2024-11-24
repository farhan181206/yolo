import cv2

cctv_url = "rtsp://admin:BengkelIT@192.168.1.64:554/Streaming/Channels/1"
cap = cv2.VideoCapture(cctv_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari CCTV.")
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
