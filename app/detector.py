
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Thư mục gốc của dự án Face_Mask
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "model" / "outputs" / "mask_detector.keras"
LABELS_PATH = PROJECT_DIR / "model" / "outputs" / "labels.json"

FACE_MODEL_PATH = PROJECT_DIR / "model" / "outputs" / "face_detector.tflite"

# Đọc đúng thứ tự nhãn đã sử dụng khi huấn luyện
if LABELS_PATH.exists():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = [
        "with_mask",
        "without_mask",
        "with_incorrect_mask",
    ]

if len(CLASS_NAMES) != 3:
    raise ValueError("Model cần đúng 3 nhãn phân loại khẩu trang.")

# Nạp model một lần khi module được khởi tạo
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy model: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# OpenCV Haar Cascade dùng để tìm khuôn mặt
# cascade_path = (
#     cv2.data.haarcascades
#     + "haarcascade_frontalface_default.xml"
# )

# face_detector = cv2.CascadeClassifier(cascade_path)

# if face_detector.empty():
#     raise RuntimeError("Không thể tải Haar Cascade face detector.")

base_options = mp.tasks.BaseOptions(
    model_asset_path=str(FACE_MODEL_PATH)
)

face_detector_options = mp.tasks.vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_detection_confidence=0.4,
)

face_detector = mp.tasks.vision.FaceDetector.create_from_options(
    face_detector_options
)

def detect_faces(image: np.ndarray) -> list[dict]:
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào không hợp lệ.")

    # MediaPipe Tasks nhận ảnh RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    detection_result = face_detector.detect(mp_image)

    predictions = []
    image_height, image_width = image.shape[:2]

    for detection in detection_result.detections:
        bbox = detection.bounding_box

        # Tọa độ khuôn mặt
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(image_width, x1 + bbox.width)
        y2 = min(image_height, y1 + bbox.height)

        if x2 <= x1 or y2 <= y1:
            continue

        # Mở rộng nhẹ vùng khuôn mặt để lấy cả khẩu trang
        margin_x = int((x2 - x1) * 0.10)
        margin_y = int((y2 - y1) * 0.10)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image_width, x2 + margin_x)
        y2 = min(image_height, y2 + margin_y)

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Tiền xử lý theo MobileNetV2
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(
            face_rgb,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        face_array = preprocess_input(
            face_resized.astype(np.float32)
        )
        face_array = np.expand_dims(face_array, axis=0)

        # Phân loại bằng model Keras đã huấn luyện
        probabilities = model.predict(
            face_array,
            verbose=0
        )[0]

        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        predictions.append({
            "bbox": {
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            },
            "label": CLASS_NAMES[class_id],
            "confidence": round(confidence, 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]), 4)
                for i in range(len(CLASS_NAMES))
            }
        })

    return predictions