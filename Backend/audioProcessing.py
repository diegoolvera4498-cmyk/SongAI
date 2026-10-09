import librosa
import numpy as np

# =========================================
# PROCESAMIENTO DE AUDIO
# =========================================

def process_audio(filepath):


    y, sr = librosa.load(filepath)


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

    # =====================================
    # MFCC
    # =====================================

    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfccs_mean = np.mean(
        mfccs.T,
        axis=0
    )

    # =====================================
    # ZERO CROSSING RATE
    # =====================================

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y)
    )

    # =====================================
    # RMS ENERGY
    # =====================================

    rms = np.mean(
        librosa.feature.rms(y=y)
    )

    # =====================================
    # SPECTRAL CENTROID
    # =====================================

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    # =====================================
    # VECTOR FINAL FEATURES
    # =====================================

    features = [
        pitch,
        zcr,
        rms,
        spectral_centroid
    ]

    features.extend(mfccs_mean)

    return features