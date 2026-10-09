from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from backend.audio.processor import AudioProcessingError, extract_audio_features

router = APIRouter()


@router.post("/analyze")
async def analyze_voice(request: Request, file: UploadFile = File(...)) -> dict:
    """Receive a WAV file, extract voice features and classify vocal tone."""

    filename = file.filename or ""
    content_type = file.content_type or ""
    if not filename.lower().endswith(".wav") and content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se aceptan archivos WAV.",
        )

    suffix = Path(filename).suffix or ".wav"
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        features = extract_audio_features(temp_path)
        model = request.app.state.vocaltone_model
        result = model.predict(features.as_vector(), pitch=features.pitch)

        return {
            "pitch": result.pitch,
            "clasificacion_vocal": result.classification,
            "confianza": result.confidence,
            "recomendacion_tono": result.recommendation,
        }
    except AudioProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrio un error interno al analizar el audio.",
        ) from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
