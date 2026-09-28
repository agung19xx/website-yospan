import os
import numpy as np

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# KONFIGURASI FLASK
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "mobilenetv2_yospan_best.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "labels.txt"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# LOAD MODEL MOBILENETV2
# ============================================================

print("==============================================")
print("MEMUAT MODEL MOBILENETV2 YOSPAN")
print("==============================================")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model berhasil dimuat.")
print("Model:", MODEL_PATH)


# ============================================================
# LOAD LABEL
# ============================================================

if os.path.exists(LABEL_PATH):

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        labels = [
            line.strip()
            for line in file
            if line.strip()
        ]

else:

    labels = [
        "bukan_yospan",
        "gale_gale",
        "jef",
        "pacul_tiga",
        "pancar",
        "seka"
    ]

print()
print("Label:", labels)
print("Jumlah kelas:", len(labels))

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

    "seka": "Seka"
}


# ============================================================
# FUNGSI MENGUBAH LABEL MENJADI NAMA YANG RAPI
# ============================================================

def format_label(label):

    key = label.strip().lower()

    return DISPLAY_NAMES.get(
        key,
        label.replace("_", " ").title()
    )


# ============================================================
# FUNGSI PREPROCESSING GAMBAR
#
# SESUAI DENGAN MODEL MOBILENETV2
#
# Input:
#   gambar
#
# Proses:
#   1. Resize menjadi 224 x 224
#   2. Ubah menjadi array
#   3. Tambahkan batch dimension
#   4. preprocess_input MobileNetV2
#
# Output:
#   Array dengan bentuk (1, 224, 224, 3)
# ============================================================

def preprocess_image(filepath):

    # --------------------------------------------------------
    # Baca gambar dan resize menjadi 224 x 224
    # --------------------------------------------------------

    image = load_img(
        filepath,
        target_size=(224, 224)
    )

    # --------------------------------------------------------
    # Ubah gambar menjadi NumPy array
    # --------------------------------------------------------

    image_array = img_to_array(
        image
    )

    # --------------------------------------------------------
    # Tambahkan dimensi batch
    #
    # Dari:
    #   (224, 224, 3)
    #
    # Menjadi:
    #   (1, 224, 224, 3)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Preprocessing khusus MobileNetV2
    # --------------------------------------------------------

    image_array = preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def predict_image(image_path):
    # Load gambar
    image = load_img(image_path, target_size=(224, 224))

    # Convert ke array
    image_array = img_to_array(image)

    # Tambahkan batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocessing MobileNetV2
    image_array = preprocess_input(image_array)

    # Prediksi
    prediction = model.predict(image_array, verbose=0)

    # Validasi jumlah output model
    if prediction.shape[1] != len(labels):
        raise ValueError(
            f"Jumlah output model ({prediction.shape[1]}) "
            f"tidak sama dengan jumlah labels ({len(labels)})"
        )

    # Ambil index dengan probabilitas tertinggi
    predicted_index = np.argmax(prediction[0])

    # Ambil label
    predicted_label = labels[predicted_index]

    # Confidence
    confidence = float(prediction[0][predicted_index])

    # Tentukan apakah Yospan
    is_yospan = predicted_label != "bukan_yospan"

    # WAJIB hanya mengembalikan 3 nilai
    return predicted_label, confidence, is_yospan


# ============================================================
# HALAMAN BERANDA
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# HALAMAN TENTANG
# ============================================================

@app.route("/tentang")
def tentang():

    return render_template(
        "tentang.html"
    )


# ============================================================
# HALAMAN KLASIFIKASI
# ============================================================

@app.route("/klasifikasi")
def klasifikasi():

    return render_template(
        "klasifikasi.html"
    )


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

        # Simpan sementara di /tmp karena Vercel hanya mengizinkan
        # penulisan pada direktori temporary
        import tempfile

        suffix = os.path.splitext(file.filename)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir="/tmp"
        ) as temp_file:

            temp_path = temp_file.name
            file.save(temp_path)

        # Prediksi
        predicted_label, confidence, is_yospan = predict_image(temp_path)

        # Baca kembali gambar untuk dikirim ke frontend
        import base64

        with open(temp_path, "rb") as image_file:
            image_base64 = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        # Hapus file sementara
        try:
            os.remove(temp_path)
        except Exception:
            pass

        # Tentukan MIME type
        mime_type = file.mimetype or "image/jpeg"

        return jsonify({
            "success": True,
            "status": "success",
            "label": predicted_label,
            "confidence": float(confidence),
            "is_yospan": is_yospan,
            "image": f"data:{mime_type};base64,{image_base64}"
        })

    except Exception as e:
        print("ERROR PREDIKSI:", str(e))

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


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
