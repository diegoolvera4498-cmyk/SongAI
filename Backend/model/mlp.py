from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from backend.model.config import FEATURE_SIZE, LABELS, RECOMMENDATIONS
from backend.model.synthetic_dataset import create_synthetic_dataset


@dataclass
class PredictionResult:
    pitch: float
    classification: str
    confidence: float
    recommendation: str


class StandardNormalizer:
    """Minimal standard-score normalizer stored alongside the Keras model."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, data: np.ndarray) -> None:
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.std = np.std(data, axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("El normalizador no ha sido entrenado.")
        return ((data - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("No se puede guardar un normalizador vacio.")
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "StandardNormalizer":
        payload = np.load(path)
        return cls(mean=payload["mean"], std=payload["std"])


class VocalToneModel:
    """Loads, trains and serves the VocalTone AI MLP classifier."""

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.storage_dir / "vocaltone_mlp.keras"
        self.normalizer_path = self.storage_dir / "normalizer.npz"
        self.normalizer = StandardNormalizer()
        self.model: keras.Model | None = None

    def load_or_train(self) -> None:
        if self.model_path.exists() and self.normalizer_path.exists():
            self.model = keras.models.load_model(self.model_path)
            self.normalizer = StandardNormalizer.load(self.normalizer_path)
            return

        features, labels = create_synthetic_dataset()
        self.normalizer.fit(features)
        normalized_features = self.normalizer.transform(features)

        self.model = self._build_model()
        self.model.fit(
            normalized_features,
            keras.utils.to_categorical(labels, num_classes=len(LABELS)),
            epochs=45,
            batch_size=32,
            verbose=0,
            validation_split=0.15,
        )
        self.model.save(self.model_path)
        self.normalizer.save(self.normalizer_path)

    def predict(self, feature_vector: np.ndarray, pitch: float) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("El modelo no esta inicializado.")

        feature_vector = feature_vector.reshape(1, FEATURE_SIZE).astype(np.float32)
        normalized = self.normalizer.transform(feature_vector)
        probabilities = self.model.predict(normalized, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        classification = LABELS[class_index]

        return PredictionResult(
            pitch=round(float(pitch), 2),
            classification=classification,
            confidence=round(float(probabilities[class_index]), 4),
            recommendation=RECOMMENDATIONS[classification],
        )

    @staticmethod
    def _build_model() -> keras.Model:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(FEATURE_SIZE,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.15),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(len(LABELS), activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
