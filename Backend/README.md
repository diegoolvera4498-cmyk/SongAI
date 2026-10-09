# VocalTone AI Backend

Backend en Python y FastAPI para analizar archivos WAV, extraer caracteristicas vocales y clasificar el tono en `grave`, `media` o `aguda`.

## Estructura

```text
backend/
  api/
    routes.py
  audio/
    processor.py
  model/
    config.py
    mlp.py
    synthetic_dataset.py
  main.py
  requirements.txt
```

## Instalacion

Desde la raiz del proyecto:

```bash
python -m venv backend\.venv
backend\.venv\Scripts\activate
pip install -r backend\requirements.txt
```

## Ejecutar localmente

```bash
uvicorn backend.main:app --reload
```

La API quedara disponible en:

```text
http://127.0.0.1:8000
```

## Endpoint principal

```http
POST /analyze
```

Recibe un archivo `WAV` como `multipart/form-data` en el campo `file`.

Ejemplo con curl:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" -F "file=@voz.wav;type=audio/wav"
```

Respuesta esperada:

```json
{
  "pitch": 221.37,
  "clasificacion_vocal": "media",
  "confianza": 0.9831,
  "recomendacion_tono": "Tonalidades sugeridas: G3-E4. Prueba Sol, La o Si menor para mantener comodidad y brillo equilibrado."
}
```

## Modelo

Si no existe un modelo entrenado en `backend/model/artifacts`, el backend crea automaticamente un dataset sintetico y entrena una red neuronal multicapa:

- Entrada: pitch, intensidad y 13 MFCCs.
- Capas ocultas: Dense con ReLU.
- Salida: Dense con Softmax para `grave`, `media`, `aguda`.
- Normalizacion: media y desviacion estandar guardadas en `normalizer.npz`.

El dataset sintetico es solo una base de desarrollo. Para produccion conviene reemplazarlo por grabaciones reales etiquetadas por rango vocal.
