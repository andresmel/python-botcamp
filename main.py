# main.py
import base64
from typing import Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from sqlalchemy import create_engine, text
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API YOLO + Supabase - Bootcamp")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 1. CONEXIÓN A SUPABASE (POSTGRES - SESSION POOLER)
# ==========================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL no está configurada")
engine = create_engine(DATABASE_URL)

# ==========================================================
# 2. CARGAR MODELO YOLO (UNA SOLA VEZ)
# ==========================================================

try:
    # Pon aquí el nombre correcto de tu modelo de caras
    model = YOLO("yolov8n-face.pt")
except Exception:
    # Fallback a un modelo general si el de caras no está
    model = YOLO("yolov8n.pt")

# ==========================================================
# 3. MODELOS Pydantic (REQUEST / RESPONSE)
# ==========================================================


class DetectRequest(BaseModel):
    image_base64: str  # imagen original enviada por el front


class DetectResponse(BaseModel):
    faces_count: int
    image_base64_result: str  # imagen procesada (con rectángulos) en base64


# ==========================================================
# 4. FUNCIONES AUXILIARES
# ==========================================================


def decode_base64_image(image_b64: str) -> np.ndarray:
    """
    Recibe un base64 (con o sin 'data:image/...;base64,')
    y devuelve una imagen OpenCV en formato BGR.
    """
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen base64 inválida")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    return img


def encode_image_to_base64(img_bgr: np.ndarray) -> str:
    """
    Convierte una imagen BGR (OpenCV) a base64 (PNG) lista para <img src="...">
    """
    ok, buffer = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la imagen")

    img_bytes = buffer.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def detectar_caras_img(img_bgr: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Recibe una imagen BGR, detecta caras y dibuja rectángulos.
    Devuelve:
      - número de caras detectadas
      - imagen BGR con las caras dibujadas
    """
    results = model(img_bgr)
    img_draw = img_bgr.copy()
    caras_detectadas = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            if confidence >= 0.5:
                caras_detectadas += 1
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"cara {caras_detectadas} ({confidence:.2f})"
                cv2.putText(
                    img_draw,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    return caras_detectadas, img_draw


def guardar_imagen_en_bd(img_base64: str) -> bool:
    """
    Guarda la imagen procesada (base64) en la tabla 'imagenes'
    de la BD 'postgres' (proyecto Supabase).
    Columna: imagen
    """
    try:
        with engine.begin() as conn:
            query = text("INSERT INTO imagenes (imagen) VALUES (:img)")
            result = conn.execute(query, {"img": img_base64})
            if result.rowcount != 1:
                print("Error al guardar en BD: No se insertó ninguna fila")
                return False

            return True
    except Exception as e:
        print("Error al guardar en BD:", e)
        return False
       


# ==========================================================
# 5. APP FASTAPI Y ENDPOINT
# ==========================================================

@app.get("/get-images")
def get_images():
  
    try:
        with engine.connect() as conn:
            query = text("SELECT * from imagenes ORDER BY id DESC")
            result = conn.execute(query)

            rows = result.mappings().all()   # convierte en dicts
            return list(rows)                # devuelve lista JSON

    except Exception as e:
        print("Error al obtener imágenes:", e)
        return []


@app.post("/detect-faces")
def detect_faces(req: DetectRequest):
    """
    1. Recibe una imagen en base64 desde el front.
    2. La decodifica y procesa con YOLO (detección de caras).
    3. Convierte la imagen procesada a base64.
    4. Guarda la imagen procesada en la tabla 'imagenes'.
    5. Devuelve número de caras + imagen procesada.
    """
    # 1. Decodificar base64 a imagen BGR
    img_bgr = decode_base64_image(req.image_base64)

    # 2. Detectar caras
    faces_count, img_result = detectar_caras_img(img_bgr)

    # 3. Codificar la imagen procesada a base64
    img_b64_result = encode_image_to_base64(img_result)

    # 4. Guardar en BD
    response=guardar_imagen_en_bd(img_b64_result)

    # 5. Respuesta al frontend
    return response


# ==========================================================
# 6. EJECUCIÓN LOCAL
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
