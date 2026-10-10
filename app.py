import base64
import io
import os
import sys
import tempfile
import threading
import types

import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify

# ------------------------------------------------------------
# Perbaikan untuk Vercel / server tanpa pustaka grafis:
# `import mediapipe` otomatis mengimpor OpenCV (cv2), dan cv2 butuh
# libGL.so.1 yang tidak ada di Vercel. Kita tidak memakai fitur
# OpenCV, jadi jika gagal diimpor cukup diganti modul kosong.
# (Di komputer sendiri cv2 normal, jadi tidak ada yang berubah.)
# ------------------------------------------------------------
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from tensorflow.keras.models import load_model


# ============================================================
# KONFIGURASI FLASK
# ============================================================

app = Flask(__name__)

# Batas ukuran unggahan: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "yospan_pose_classifier_final.keras")
POSE_TASK_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
SCALER_PATH = os.path.join(MODEL_DIR, "pose_scaler.npz")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.txt")

# Urutan kelas HARUS sama dengan LabelEncoder saat training
# (urutan alfabet). Dipakai jika labels.txt tidak ditemukan.
DEFAULT_LABELS = [
    "bukan_yospan",
    "gale_gale",
    "jef",
    "pacul_tiga",
    "pancar",
    "seka",
]

NUM_LANDMARKS = 33
NUM_FEATURES = NUM_LANDMARKS * 4  # x, y, z, visibility = 132


# ============================================================
# LOAD MODEL KLASIFIKASI (MLP BERBASIS POSE)
# ============================================================

print("==============================================")
print("MEMUAT MODEL KLASIFIKASI POSE YOSPAN")
print("==============================================")

model = load_model(MODEL_PATH, compile=False)

print("Model berhasil dimuat:", MODEL_PATH)
print("Input model :", model.input_shape)
print("Output model:", model.output_shape)


# ============================================================
# LOAD LABEL
# ============================================================

if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file if line.strip()]
else:
    labels = list(DEFAULT_LABELS)

if model.output_shape[-1] != len(labels):
    raise ValueError(
        f"Jumlah output model ({model.output_shape[-1]}) "
        f"tidak sama dengan jumlah label ({len(labels)})."
    )

if model.input_shape[-1] != NUM_FEATURES:
    raise ValueError(
        f"Input model ({model.input_shape[-1]}) "
        f"tidak sama dengan jumlah fitur pose ({NUM_FEATURES})."
    )

print("Label:", labels)


# ============================================================
# LOAD STANDARD SCALER (hasil training)
#
# Disimpan sebagai mean & scale dari StandardScaler agar tidak
# bergantung pada versi scikit-learn.
# Rumus: (x - mean) / scale
# ============================================================

_scaler = np.load(SCALER_PATH)
SCALER_MEAN = _scaler["mean"].astype(np.float32)
SCALER_SCALE = _scaler["scale"].astype(np.float32)

if SCALER_MEAN.shape[0] != NUM_FEATURES:
    raise ValueError("Ukuran scaler tidak sesuai dengan jumlah fitur.")


# ============================================================
# MEDIAPIPE POSE LANDMARKER
#
# Pengaturan SAMA dengan saat ekstraksi fitur di notebook
# training (Cell 5): num_poses=1, confidence 0.5, mode IMAGE.
# ============================================================

pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_options)

# Detector dipakai bersama oleh semua request, jadi diberi lock.
pose_lock = threading.Lock()

print("MediaPipe Pose Landmarker siap.")
print("==============================================")
print("MODEL SIAP DIGUNAKAN")
print("==============================================")


# ============================================================
# NAMA GERAKAN UNTUK DITAMPILKAN DI WEBSITE
# ============================================================

DISPLAY_NAMES = {
    "bukan_yospan": "Bukan Yospan",

    "gale_gale": "Gale-gale",
    "gale-gale": "Gale-gale",
    "gale gale": "Gale-gale",

    "jef": "Jef",

    "pacul_tiga": "Pacul Tiga",
    "pacul-tiga": "Pacul Tiga",
    "pacul tiga": "Pacul Tiga",

    "pancar": "Pancar",

    "seka": "Seka",
}


def format_label(label):
    key = label.strip().lower()
    return DISPLAY_NAMES.get(key, label.replace("_", " ").title())


# ============================================================
# NORMALISASI POSE
#
# SAMA PERSIS dengan fungsi normalize_pose() di notebook
# training (Cell 5):
#   1. Titik pusat = rata-rata (pusat bahu, pusat pinggul)
#   2. x, y, z dikurangi titik pusat
#   3. x, y, z dibagi jarak antar bahu
#   4. visibility tidak diubah
# ============================================================

def normalize_pose(landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
        dtype=np.float32,
    )

    shoulder_center = (points[11, :3] + points[12, :3]) / 2.0
    hip_center = (points[23, :3] + points[24, :3]) / 2.0
    body_center = (shoulder_center + hip_center) / 2.0

    points[:, :3] -= body_center

    shoulder_distance = np.linalg.norm(points[11, :3] - points[12, :3])

    if shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    points[:, :3] /= shoulder_distance

    return points


# ============================================================
# MEMBUKA GAMBAR UNTUK MEDIAPIPE
#
# Saat training, gambar dibaca dengan mp.Image.create_from_file().
# Decoder JPEG/PNG MediaPipe sedikit berbeda dari PIL (selisih
# piksel kecil), dan landmark sangat sensitif terhadap itu.
# Maka gambar di sini dibaca dengan cara yang SAMA seperti training.
#
# PIL hanya dipakai untuk:
#   - membaca orientasi EXIF (foto kamera ponsel)
#   - cadangan jika format tidak didukung decoder MediaPipe
# ============================================================

EXIF_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def to_mediapipe_image(image_bytes, pil_image):
    orientation = pil_image.getexif().get(0x0112, 1)

    mp_image = None
    suffix = ".png" if pil_image.format == "PNG" else ".jpg"

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(image_bytes)

        try:
            mp_image = mp.Image.create_from_file(temp_path)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    except Exception:
        mp_image = None

    # Gambar normal (tanpa rotasi EXIF) dan berhasil dibaca MediaPipe
    if mp_image is not None and orientation not in EXIF_TRANSPOSE:
        return mp_image

    # Ada rotasi EXIF, atau format tidak didukung decoder MediaPipe
    if mp_image is not None:
        rgb = np.array(mp_image.numpy_view())[..., :3]
        pil_rgb = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
    else:
        pil_rgb = pil_image.convert("RGB")

    if orientation in EXIF_TRANSPOSE:
        pil_rgb = pil_rgb.transpose(EXIF_TRANSPOSE[orientation])

    rgb = np.ascontiguousarray(np.array(pil_rgb, dtype=np.uint8))

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ============================================================
# EKSTRAKSI FITUR DARI GAMBAR
#
# Return: array (1, 132) yang sudah distandardisasi,
#         atau None jika pose tidak terdeteksi.
# ============================================================

def extract_features(image_bytes, pil_image):
    mp_image = to_mediapipe_image(image_bytes, pil_image)

    with pose_lock:
        result = pose_detector.detect(mp_image)

    if len(result.pose_landmarks) == 0:
        return None

    landmarks = result.pose_landmarks[0]

    if len(landmarks) != NUM_LANDMARKS:
        return None

    features = normalize_pose(landmarks).flatten().astype(np.float32)

    # StandardScaler
    features = (features - SCALER_MEAN) / SCALER_SCALE

    return np.expand_dims(features, axis=0)


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def predict_pose(image_bytes, pil_image):
    features = extract_features(image_bytes, pil_image)

    if features is None:
        return None

    prediction = model.predict(features, verbose=0)

    predicted_index = int(np.argmax(prediction[0]))
    predicted_label = labels[predicted_index]
    confidence = float(prediction[0][predicted_index])
    is_yospan = predicted_label != "bukan_yospan"

    return predicted_label, confidence, is_yospan


# ============================================================
# HALAMAN
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tentang")
def tentang():
    return render_template("tentang.html")


@app.route("/klasifikasi")
def klasifikasi():
    return render_template("klasifikasi.html")


# ============================================================
# API / PROSES KLASIFIKASI
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "message": "Tidak ada gambar yang diunggah."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "message": "Tidak ada gambar yang dipilih."
            }), 400

        image_bytes = file.read()

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()
        except Exception:
            return jsonify({
                "success": False,
                "message": "File tidak dapat dibaca sebagai gambar. "
                           "Gunakan format JPG atau PNG."
            }), 400

        result = predict_pose(image_bytes, pil_image)

        if result is None:
            return jsonify({
                "success": False,
                "message": "Pose tubuh tidak terdeteksi pada gambar. "
                           "Gunakan foto yang menampilkan satu orang "
                           "dengan seluruh tubuh terlihat jelas."
            }), 422

        predicted_label, confidence, is_yospan = result

        mime_type = file.mimetype or "image/jpeg"
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify({
            "success": True,
            "status": "success",
            "label": format_label(predicted_label),
            "confidence": confidence,
            "is_yospan": is_yospan,
            "image": f"data:{mime_type};base64,{image_base64}"
        })

    except Exception as e:
        print("ERROR PREDIKSI:", str(e))

        return jsonify({
            "success": False,
            "message": "Terjadi kesalahan saat memproses gambar."
        }), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "message": "Ukuran gambar terlalu besar. Maksimal 10 MB."
    }), 413


# ============================================================
# MENJALANKAN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )