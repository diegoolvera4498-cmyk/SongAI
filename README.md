# Identificador de voz para canto

Frontend estatico para analizar voz cantada con espectrogramas mel y una CNN en TensorFlow.js.

## Backend VocalTone AI

Este repositorio tambien incluye un backend FastAPI en `backend/` para recibir archivos WAV, extraer pitch, MFCCs e intensidad con `librosa`, clasificar el tono vocal con un MLP en TensorFlow/Keras y devolver una recomendacion de tonalidad.

Consulta las instrucciones completas en `backend/README.md`.

## Uso

1. Sirve la carpeta con un servidor local.
2. Abre `index.html`.
3. Pulsa `Iniciar en vivo` para detectar desde el microfono en tiempo real.
4. Tambien puedes cargar un archivo de audio y pulsar `Analizar`.

La app intenta cargar automaticamente un modelo entrenado desde:

```text
models/voice-tone/model.json
```

Tambien puedes cargar manualmente `model.json` junto con sus archivos `.bin` desde el boton `Cargar modelo CNN`.

## Entrada esperada del modelo

La CNN recibe tensores con forma:

```text
[batch, 64, 96, 1]
```

Cada muestra es un espectrograma mel normalizado. Las clases de salida son:

```text
Agudo claro
Medio brillante
Medio calido
Grave profundo
```

Si no hay pesos entrenados, el frontend crea la arquitectura CNN y combina su salida con un perfil acustico demo para que el flujo sea probable y visible durante desarrollo.

## Deteccion en vivo

El modo en vivo usa Web Audio API con un buffer circular de `3.5` segundos. Cada `800 ms` toma la ventana mas reciente, calcula el espectrograma mel y actualiza la salida de la CNN. Para canto real conviene entrenar el modelo con ejemplos etiquetados en ventanas de duracion similar.
