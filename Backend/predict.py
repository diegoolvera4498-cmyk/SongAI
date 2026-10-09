import numpy as np
import joblib

from tensorflow.keras.models import load_model


model = load_model("models/voice_model.h5")


scaler = joblib.load(
    "models/scaler.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

def predict_voice(features):

    # =============================
    # CONVERTIR A ARRAY
    # =============================

    X = np.array(features).reshape(1, -1)

    # =============================
    # NORMALIZAR
    # =============================

    X_scaled = scaler.transform(X)


    prediction = model.predict(X_scaled)


    predicted_class = np.argmax(prediction)

    # =============================
    # TRADUCIR LABEL
    # =============================

    label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = float(np.max(prediction))

    probabilities = prediction[0].tolist()

    return {
    "voice_type": label,
    "confidence": confidence,
    "probabilities": probabilities
    }