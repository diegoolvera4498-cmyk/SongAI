from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import medfilt

from backend.model.config import FEATURE_SIZE, N_MFCC, SAMPLE_RATE


@dataclass(frozen=True)
class AudioFeatures:
    """Numerical descriptors used by the MLP classifier."""

    pitch: float
    intensity: float
    mfcc_mean: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array([self.pitch, self.intensity], dtype=np.float32),
                self.mfcc_mean.astype(np.float32),
            ]
        )


class AudioProcessingError(ValueError):
    """Raised when an audio file cannot be converted into usable features."""


def extract_audio_features(audio_path: str | Path) -> AudioFeatures:
    """Load a WAV file and extract pitch, intensity and MFCC descriptors."""

    try:
        signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # librosa can raise several decoder-specific errors.
        raise AudioProcessingError("No se pudo leer el archivo WAV enviado.") from exc

    if signal.size == 0 or float(np.max(np.abs(signal))) < 1e-6:
        raise AudioProcessingError("El archivo no contiene una senal de voz detectable.")

    # Trim long silences so pitch and MFCCs focus on the sung/voiced region.
    signal, _ = librosa.effects.trim(signal, top_db=30)
    if signal.size < sample_rate // 4:
        raise AudioProcessingError("El audio es demasiado corto para analizar la voz.")

    pitch = _estimate_pitch(signal, sample_rate)
    intensity = _estimate_intensity(signal)
    mfcc_mean = _estimate_mfcc(signal, sample_rate)

    vector = np.concatenate(([pitch, intensity], mfcc_mean))
    if vector.shape[0] != FEATURE_SIZE or not np.all(np.isfinite(vector)):
        raise AudioProcessingError("No se pudieron extraer caracteristicas validas del audio.")

    return AudioFeatures(pitch=pitch, intensity=intensity, mfcc_mean=mfcc_mean)


def _estimate_pitch(signal: np.ndarray, sample_rate: int) -> float:
    """Estimate the fundamental frequency in Hz using voiced frames."""

    try:
        f0, voiced_flag, _ = librosa.pyin(
            signal,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sample_rate,
        )
        voiced_f0 = f0[voiced_flag]
        voiced_f0 = voiced_f0[np.isfinite(voiced_f0)]
    except Exception:
        voiced_f0 = np.array([], dtype=np.float32)

    if voiced_f0.size == 0:
        fallback = librosa.yin(
            signal,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sample_rate,
        )
        voiced_f0 = fallback[np.isfinite(fallback)]

    if voiced_f0.size == 0:
        raise AudioProcessingError("No se detecto pitch vocal en el audio.")

    if voiced_f0.size >= 5:
        voiced_f0 = medfilt(voiced_f0, kernel_size=5)

    return float(np.median(voiced_f0))


def _estimate_intensity(signal: np.ndarray) -> float:
    rms = librosa.feature.rms(y=signal)[0]
    return float(np.mean(rms))


def _estimate_mfcc(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1).astype(np.float32)
