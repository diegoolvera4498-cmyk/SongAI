import os
import librosa
import numpy as np
import pandas as pd


DATASET_PATH = "../data/FULL"


data = []

# =========================================
# ETIQUETAS SIMPLES
# =========================================

def get_label(folder_name):

    folder_name = folder_name.lower()

    if "female" in folder_name:
        return "Aguda"

    elif "male" in folder_name:
        return "Grave"

    return "Desconocido"

# =========================================
# RECORRER DATASET
# =========================================

for singer_folder in os.listdir(DATASET_PATH):

    singer_path = os.path.join(DATASET_PATH, singer_folder)

    if not os.path.isdir(singer_path):
        continue

    # =====================================
    # SOLO LONG_TONES
    # =====================================

    long_tones_path = os.path.join(singer_path, "long_tones")

    if not os.path.exists(long_tones_path):
        continue

    # =====================================
    # RECORRER ARCHIVOS
    # =====================================

    for root, dirs, files in os.walk(long_tones_path):

        for file in files:

            if not file.endswith(".wav"):
                continue

            filepath = os.path.join(root, file)

            print(f"Procesando: {filepath}")

            try:

                # =============================
                # CARGAR AUDIO
                # =============================

                y, sr = librosa.load(filepath)

                # =============================
                # PITCH
                # =============================

                pitches, magnitudes = librosa.piptrack(
                    y=y,
                    sr=sr
                )

                pitch_values = pitches[
                    (pitches > 50) & (pitches < 500)
                ]

                if len(pitch_values) > 0:
                    pitch = np.mean(pitch_values)
                else:
                    pitch = 0

                # =============================
                # MFCC
                # =============================

                mfccs = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_mfcc=13
                )

                mfccs_mean = np.mean(mfccs.T, axis=0)

                # =============================
                # ZCR
                # =============================

                zcr = np.mean(
                    librosa.feature.zero_crossing_rate(y)
                )

                # =============================
                # RMS ENERGY
                # =============================

                rms = np.mean(
                    librosa.feature.rms(y=y)
                )

                # =============================
                # SPECTRAL CENTROID
                # =============================

                spectral_centroid = np.mean(
                    librosa.feature.spectral_centroid(
                        y=y,
                        sr=sr
                    )
                )

                # =============================
                # LABEL
                # =============================

                label = get_label(singer_folder)

                # =============================
                # VECTOR FINAL
                # =============================

                row = [
                    pitch,
                    zcr,
                    rms,
                    spectral_centroid
                ]

                row.extend(mfccs_mean)

                row.append(label)

                data.append(row)

            except Exception as e:

                print(f"Error en {filepath}")
                print(e)


columns = [
    "pitch",
    "zcr",
    "rms",
    "spectral_centroid"
]

for i in range(13):
    columns.append(f"mfcc_{i+1}")

columns.append("label")


df = pd.DataFrame(data, columns=columns)


df.to_csv("voice_dataset.csv", index=False)

print("\nDataset creado correctamente ")
print(df.head())