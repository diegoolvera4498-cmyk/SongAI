from __future__ import annotations

import numpy as np

from backend.model.config import LABELS, N_MFCC


def create_synthetic_dataset(samples_per_class: int = 250, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create a small feature dataset when no labeled voice dataset exists.

    The synthetic data approximates three vocal ranges through pitch bands,
    energy variation and MFCC-like descriptors. It is useful for local demos
    and should be replaced by real labeled recordings for production.
    """

    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []

    class_specs = {
        0: {"pitch": (80, 175), "intensity": (0.035, 0.14), "mfcc_center": -22.0},
        1: {"pitch": (176, 330), "intensity": (0.045, 0.18), "mfcc_center": -12.0},
        2: {"pitch": (331, 720), "intensity": (0.04, 0.16), "mfcc_center": -4.0},
    }

    for class_index in range(len(LABELS)):
        spec = class_specs[class_index]
        low_pitch, high_pitch = spec["pitch"]
        low_intensity, high_intensity = spec["intensity"]

        for _ in range(samples_per_class):
            pitch = rng.uniform(low_pitch, high_pitch)
            intensity = rng.uniform(low_intensity, high_intensity)

            # MFCCs are correlated lightly with class and pitch to make the MLP
            # use more than a hard pitch threshold during training.
            mfcc_base = np.linspace(spec["mfcc_center"], spec["mfcc_center"] + 10, N_MFCC)
            mfcc_noise = rng.normal(0, 3.2, N_MFCC)
            mfcc_pitch_bias = np.log1p(pitch) * rng.normal(0.15, 0.03, N_MFCC)
            mfcc = mfcc_base + mfcc_noise + mfcc_pitch_bias

            sample = np.concatenate(([pitch, intensity], mfcc)).astype(np.float32)
            features.append(sample)
            labels.append(class_index)

    return np.vstack(features).astype(np.float32), np.array(labels, dtype=np.int64)
