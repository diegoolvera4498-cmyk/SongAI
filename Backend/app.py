from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from audioProcessing import process_audio
from predict import predict_voice

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================================
# RUTA PRINCIPAL
# =========================================

@app.route("/")
def home():

    return "SONGAI Backend funcionando 🎵"


@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:

        return jsonify({
            "error": "No se encontró audio"
        }), 400


    audio = request.files["audio"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        audio.filename
    )

    audio.save(filepath)

    # =====================================
    # EXTRAER FEATURES
    # =====================================

    features = process_audio(filepath)

    # =====================================
    # IA PREDICCIÓN
    # =====================================

    result = predict_voice(features)



    return jsonify(result)

# =========================================

if __name__ == "__main__":

    app.run(debug=True)