# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import HTMLResponse
# from pathlib import Path
# import cv2
# import numpy as np
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse

# WEB_DIR = Path(__file__).parent / "static"
# app = FastAPI(title="Mask Detection Cloud API")
# app.mount(
#     "/static",
#     StaticFiles(directory=WEB_DIR),
#     name="static"
# )

# @app.get("/", include_in_schema=False)
# def dashboard():
#     return FileResponse(WEB_DIR / "index.html")

# WEB_DIR = Path(__file__).parent / "static"

# @app.get("/api/health")
# def health():
#     return {"status": "ok", "service": "mask-detection"}

# @app.post("/api/detect")
# async def detect(file: UploadFile = File(...)):
#     if file.content_type not in ("image/jpeg", "image/png"):
#         raise HTTPException(400, "Only JPEG and PNG images are accepted")

#     raw = await file.read()

#     if len(raw) > 2_000_000:
#         raise HTTPException(413, "Image too large")

#     image = cv2.imdecode(
#         np.frombuffer(raw, dtype=np.uint8),
#         cv2.IMREAD_COLOR
#     )

#     if image is None:
#         raise HTTPException(400, "Invalid image")

#     # TODO: Gọi face detector và MobileNetV2 tại đây.
#     # Chưa có model thì chưa thể trả nhãn AI thực tế.

#     return {
#         "status": "received",
#         "width": int(image.shape[1]),
#         "height": int(image.shape[0]),
#         "predictions": []
#     }

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.detector import detect_faces


WEB_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model đã được nạp khi import app.detector.
    print("[INFO] Mask detection API is ready.")
    yield


app = FastAPI(
    title="Mask Detection Cloud API",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR),
    name="static",
)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "mask-detection",
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type not in (
        "image/jpeg",
        "image/png",
    ):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are accepted.",
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Empty image.",
        )

    if len(raw) > 2_000_000:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Maximum size is 2 MB.",
        )

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    try:
        predictions = detect_faces(image)
    except Exception as exc:
        print(f"[ERROR] Detection failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Face detection or model inference failed.",
        ) from exc

    return {
        "status": "success",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "face_count": len(predictions),
        "predictions": predictions,
    }
