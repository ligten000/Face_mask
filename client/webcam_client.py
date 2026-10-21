import cv2
import requests
import time

API_URL = "http://127.0.0.1:8000/api/detect"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Cannot read webcam frame")
            break

        # Giảm kích thước ảnh để giảm lưu lượng mạng
        frame = cv2.resize(frame, (640, 480))

        ok, encoded = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if ok:
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            "frame.jpg",
                            encoded.tobytes(),
                            "image/jpeg"
                        )
                    },
                    timeout=5
                )
                print(response.status_code, response.text)
            except requests.RequestException as exc:
                print("API error:", exc)

        cv2.imshow("USB Webcam", frame)

        # Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Giới hạn tốc độ gửi ảnh trong giai đoạn thử nghiệm
        time.sleep(0.5)

finally:
    cap.release()
    cv2.destroyAllWindows()