"""Shared model configuration."""

LABELS = ("grave", "media", "aguda")

RECOMMENDATIONS = {
    "grave": "Tonalidades sugeridas: C2-G3. Prueba canciones en Do, Re o Mi menor para aprovechar cuerpo y estabilidad.",
    "media": "Tonalidades sugeridas: G3-E4. Prueba Sol, La o Si menor para mantener comodidad y brillo equilibrado.",
    "aguda": "Tonalidades sugeridas: C4-A4. Prueba Do, Re o Mi mayor para resaltar claridad y proyeccion.",
}

SAMPLE_RATE = 22050
N_MFCC = 13
FEATURE_SIZE = 15
