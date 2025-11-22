# main.py
import base64
from typing import Tuple
from dotenv import load_dotenv
load_dotenv()
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from sqlalchemy import create_engine, text
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

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
# DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL no está configurada")
engine = create_engine(DATABASE_URL)

# ==========================================================
# 2. CARGAR MODELOS YOLO (CARAS + OBJETOS)
# ==========================================================
# Modelo de caras
try:
    face_model = YOLO("yolov8n-face.pt")
except Exception:
    face_model = None

# Modelo general de objetos
try:
    obj_model = YOLO("yolov8n.pt")
except Exception:
    obj_model = None

# ==========================================================
# 3. MODELOS Pydantic (REQUEST / RESPONSE)
# ==========================================================

class DetectRequest(BaseModel):
    image_base64: str  # imagen original enviada por el front

class DetectResponse(BaseModel):
    faces_count: int
    image_base64_result: str  # imagen procesada (con rectángulos) en base64
    description: str

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
    Detecta caras (si hay modelo de caras) y dibuja rectángulos.
    Devuelve:
      - número de caras detectadas
      - imagen BGR con las caras dibujadas
    """
    img_draw = img_bgr.copy()
    if face_model is None:
        return 0, img_draw

    results = face_model(img_bgr)
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

# --------- Heurísticas para descripción ----------
def _estimate_brightness(img_bgr: np.ndarray) -> str:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_mean = float(hsv[..., 2].mean())
    if v_mean >= 180:
        return "muy iluminada"
    elif v_mean >= 120:
        return "bien iluminada"
    elif v_mean >= 80:
        return "con luz media"
    else:
        return "con poca luz"

def _estimate_sharpness(img_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm >= 200.0:
        return "nítida"
    elif fm >= 80.0:
        return "ligeramente borrosa"
    else:
        return "borrosa"

def _orientation(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    if h > w * 1.1:
        return "vertical"
    elif w > h * 1.1:
        return "horizontal"
    else:
        return "cuadrada"

def build_auto_description_from_yolo(img_bgr: np.ndarray, faces_count: int) -> str:
    """
    Usa el modelo general para listar objetos/personas y arma una descripción natural.
    También incluye condiciones de luz, nitidez y orientación.
    """
    bright_txt = _estimate_brightness(img_bgr)
    sharp_txt  = _estimate_sharpness(img_bgr)
    orient_txt = _orientation(img_bgr)

    # Conteo de objetos
    obj_counts = {}
    if obj_model is not None:
        results = obj_model(img_bgr)
        for r in results:
            names = r.names if hasattr(r, "names") else getattr(obj_model.model, "names", {})
            if len(r.boxes):
                for cls_id in r.boxes.cls.tolist():
                    cls_id = int(cls_id)
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    obj_counts[label] = obj_counts.get(label, 0) + 1

    people = obj_counts.get("person", 0)

    partes = []
    # 1) Caras / personas
    if faces_count > 0:
        partes.append(f"Se detectaron {faces_count} rostro(s).")
    elif people > 0:
        partes.append(f"Se detectaron {people} persona(s) de cuerpo completo.")
    else:
        partes.append("No se detectaron rostros ni personas claramente.")

    # 2) Otros objetos (top 5 distintos a person)
    otros = [(k, v) for k, v in obj_counts.items() if k != "person"]
    otros.sort(key=lambda kv: kv[1], reverse=True)
    if otros:
        top = ", ".join([f"{k} ({v})" for k, v in otros[:5]])
        partes.append(f"Objetos presentes: {top}.")

    # 3) Condiciones de imagen
    partes.append(f"Imagen {bright_txt}, {sharp_txt}, orientación {orient_txt}.")

    # 4) Timestamp simple
    partes.append(datetime.now().strftime("Procesada el %Y-%m-%d %H:%M:%S."))

    return " ".join(partes)

def guardar_imagen_en_bd(img_base64: str, description: str) -> bool:
    """
    Guarda la imagen procesada y la descripción en la tabla 'imagenes'
    (columnas: imagen, description)
    """
    try:
        with engine.begin() as conn:
            query = text("""
                INSERT INTO imagenes (imagen, descroption)
                VALUES (:img, :desc)
            """)
            result = conn.execute(query, {"img": img_base64, "desc": description})
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
    3. Genera una descripción automática con YOLO+heurísticas.
    4. Convierte la imagen procesada a base64.
    5. Guarda imagen + descripción en la BD.
    6. Devuelve número de caras + imagen + descripción.
    """
    # 1. Decodificar base64 a imagen BGR
    img_bgr = decode_base64_image(req.image_base64)

    # 2. Detectar caras
    faces_count, img_result = detectar_caras_img(img_bgr)

    # 3. Descripción automática
    description = build_auto_description_from_yolo(img_bgr, faces_count)

    # 4. Codificar imagen procesada a base64
    img_b64_result = encode_image_to_base64(img_result)

    # 5. Guardar en BD
    ok = guardar_imagen_en_bd(img_b64_result, description)
    return ok
    

# ==========================================================
# 6. EJECUCIÓN LOCAL
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
