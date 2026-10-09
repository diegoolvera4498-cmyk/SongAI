const LABELS = [
  {
    name: "Agudo claro",
    advice: "Usa un tono ligero, frontal y brillante. Funciona bien para melodias altas, pop latino y coros con mucha presencia.",
  },
  {
    name: "Medio brillante",
    advice: "Elige un tono flexible con ataque definido. Es una buena base para balada, pop y frases melodicas sostenidas.",
  },
  {
    name: "Medio calido",
    advice: "Conviene un tono redondo y cercano, con menos aire y mas cuerpo. Ideal para bolero, indie, R&B suave y partes intimas.",
  },
  {
    name: "Grave profundo",
    advice: "Trabaja con un tono oscuro, estable y apoyado. Favorece versos graves, narrativos y arreglos con poca saturacion.",
  },
];

const MEL_BINS = 64;
const TIME_STEPS = 96;
const FFT_SIZE = 512;
const HOP_SIZE = 256;
const LIVE_WINDOW_SECONDS = 3.5;
const LIVE_PREDICTION_INTERVAL_MS = 800;

let audioBuffer = null;
let model = null;
let modelReady = false;
let lastFeatures = null;
let liveContext = null;
let liveStream = null;
let liveSource = null;
let liveProcessor = null;
let liveSamples = null;
let liveWriteIndex = 0;
let liveSamplesWritten = 0;
let livePredictTimer = null;
let liveIsRunning = false;
let liveIsPredicting = false;

const $ = (id) => document.getElementById(id);

let ui = null;
let modelInitPromise = null;

function initializeUI() {
  ui = {
    audioFile: $("audioFile"),
    modelFiles: $("modelFiles"),
    recordButton: $("recordButton"),
    analyzeButton: $("analyzeButton"),
    modelBadge: $("modelBadge"),
    modeLabel: $("modeLabel"),
    audioMeta: $("audioMeta"),
    spectrogram: $("spectrogram"),
    confidenceLabel: $("confidenceLabel"),
    confidenceBar: $("confidenceBar"),
    primaryLabel: $("primaryLabel"),
    recommendation: $("recommendation"),
    pitchValue: $("pitchValue"),
    centroidValue: $("centroidValue"),
    energyValue: $("energyValue"),
    probabilities: $("probabilities"),
  };

  setupEventListeners();
  modelInitPromise = init();
}

async function init() {
  renderProbabilities(new Array(LABELS.length).fill(0));
  drawEmptySpectrogram();
  await waitForTensorFlow();
  model = createVoiceToneCnn();
  ui.modelBadge.textContent = "CNN demo lista";
  ui.modeLabel.textContent = "CNN demo";
  modelReady = true;

  try {
    model = await tf.loadLayersModel("models/voice-tone/model.json");
    ui.modelBadge.textContent = "Modelo CNN cargado";
    ui.modeLabel.textContent = "CNN entrenada";
  } catch {
    ui.modelBadge.textContent = "CNN demo lista";
  }
}

function setupEventListeners() {
  ui.audioFile.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await loadAudioFile(file);
  });

  ui.modelFiles.addEventListener("change", async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;

    try {
      model = await tf.loadLayersModel(tf.io.browserFiles(files));
      ui.modelBadge.textContent = "Modelo CNN cargado";
      ui.modeLabel.textContent = "CNN entrenada";
    } catch (error) {
      ui.modelBadge.textContent = "Modelo invalido";
      console.error(error);
    }
  });

  ui.recordButton.addEventListener("click", async () => {
    if (liveIsRunning) {
      stopLiveDetection();
      return;
    }

    try {
      await startLiveDetection();
    } catch (error) {
      ui.modelBadge.textContent = "Microfono no disponible";
      console.error(error);
    }
  });

  ui.analyzeButton.addEventListener("click", async () => {
    if (!audioBuffer || !modelReady) return;
    setBusy(true);
    await new Promise((resolve) => setTimeout(resolve, 20));

    try {
      const mono = mixToMono(audioBuffer);
      lastFeatures = extractFeatures(mono, audioBuffer.sampleRate);
      drawSpectrogram(lastFeatures.melSpectrogram);

      const prediction = await predictVoiceTone(lastFeatures);
      renderResult(prediction, lastFeatures.stats);
    } finally {
      setBusy(false);
    }
  });
}

async function loadAudioFile(file) {
  stopLiveDetection();
  const context = new AudioContext();
  const bytes = await file.arrayBuffer();
  audioBuffer = await context.decodeAudioData(bytes);
  ui.analyzeButton.disabled = false;
  ui.audioMeta.textContent = `${formatSeconds(audioBuffer.duration)} - ${Math.round(audioBuffer.sampleRate / 1000)} kHz`;
  ui.primaryLabel.textContent = "Audio listo";
  ui.recommendation.textContent = "Pulsa Analizar para crear el espectrograma y estimar el tono vocal.";
}

async function startLiveDetection() {
  if (!modelReady) {
    ui.modelBadge.textContent = "CNN inicializando";
    await modelInitPromise;
  }

  liveContext = new AudioContext();
  if (liveContext.state === "suspended") {
    await liveContext.resume();
  }
  liveStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  liveSource = liveContext.createMediaStreamSource(liveStream);
  liveProcessor = liveContext.createScriptProcessor(4096, 1, 1);
  liveSamples = new Float32Array(Math.floor(liveContext.sampleRate * LIVE_WINDOW_SECONDS));
  liveWriteIndex = 0;
  liveSamplesWritten = 0;

  liveProcessor.onaudioprocess = (event) => {
    writeLiveSamples(event.inputBuffer.getChannelData(0));
    event.outputBuffer.getChannelData(0).fill(0);
  };

  liveSource.connect(liveProcessor);
  liveProcessor.connect(liveContext.destination);
  liveIsRunning = true;
  audioBuffer = null;
  ui.analyzeButton.disabled = true;
  ui.recordButton.textContent = "Detener en vivo";
  ui.recordButton.classList.add("is-recording");
  ui.audioMeta.textContent = "Escuchando en vivo";
  ui.primaryLabel.textContent = "Detectando...";
  ui.recommendation.textContent = "Canta o habla cerca del microfono para actualizar el tono en tiempo real.";

  livePredictTimer = window.setInterval(runLivePrediction, LIVE_PREDICTION_INTERVAL_MS);
}

function stopLiveDetection() {
  if (!liveIsRunning) return;

  window.clearInterval(livePredictTimer);
  livePredictTimer = null;
  liveProcessor?.disconnect();
  liveSource?.disconnect();
  liveStream?.getTracks().forEach((track) => track.stop());
  liveContext?.close();

  liveContext = null;
  liveStream = null;
  liveSource = null;
  liveProcessor = null;
  liveSamples = null;
  liveIsRunning = false;
  liveIsPredicting = false;
  ui.recordButton.textContent = "Iniciar en vivo";
  ui.recordButton.classList.remove("is-recording");
  ui.audioMeta.textContent = audioBuffer
    ? `${formatSeconds(audioBuffer.duration)} - ${Math.round(audioBuffer.sampleRate / 1000)} kHz`
    : "Sin audio";
  ui.analyzeButton.disabled = !audioBuffer;
}

function writeLiveSamples(input) {
  for (let i = 0; i < input.length; i++) {
    liveSamples[liveWriteIndex] = input[i];
    liveWriteIndex = (liveWriteIndex + 1) % liveSamples.length;
  }
  liveSamplesWritten = Math.min(liveSamples.length, liveSamplesWritten + input.length);
}

async function runLivePrediction() {
  if (!liveIsRunning || liveIsPredicting || !modelReady || liveSamplesWritten < FFT_SIZE * 4) return;

  const samples = readLiveWindow();
  const rms = Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
  if (rms < 0.01) {
    ui.primaryLabel.textContent = "Esperando voz";
    ui.confidenceLabel.textContent = "0%";
    ui.confidenceBar.style.width = "0%";
    ui.energyValue.textContent = `${Math.round(rms * 1000) / 1000}`;
    return;
  }

  liveIsPredicting = true;
  try {
    lastFeatures = extractFeatures(samples, liveContext.sampleRate);
    drawSpectrogram(lastFeatures.melSpectrogram);
    const prediction = await predictVoiceTone(lastFeatures);
    renderResult(prediction, lastFeatures.stats);
    ui.audioMeta.textContent = `En vivo - ${Math.round(liveContext.sampleRate / 1000)} kHz`;
  } finally {
    liveIsPredicting = false;
  }
}

function readLiveWindow() {
  const length = Math.min(liveSamplesWritten, liveSamples.length);
  const output = new Float32Array(length);
  const start = (liveWriteIndex - length + liveSamples.length) % liveSamples.length;

  for (let i = 0; i < length; i++) {
    output[i] = liveSamples[(start + i) % liveSamples.length];
  }

  return output;
}

async function waitForTensorFlow() {
  while (!window.tf) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

function createVoiceToneCnn() {
  const cnn = tf.sequential({
    layers: [
      tf.layers.inputLayer({ inputShape: [MEL_BINS, TIME_STEPS, 1] }),
      tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: "relu", padding: "same" }),
      tf.layers.maxPooling2d({ poolSize: [2, 2] }),
      tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu", padding: "same" }),
      tf.layers.maxPooling2d({ poolSize: [2, 2] }),
      tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu", padding: "same" }),
      tf.layers.globalAveragePooling2d(),
      tf.layers.dense({ units: 32, activation: "relu" }),
      tf.layers.dropout({ rate: 0.25 }),
      tf.layers.dense({ units: LABELS.length, activation: "softmax" }),
    ],
  });

  cnn.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return cnn;
}

async function predictVoiceTone(features) {
  const input = tf.tensor4d(features.melSpectrogram, [1, MEL_BINS, TIME_STEPS, 1]);
  const cnnScores = Array.from(await model.predict(input).data());
  input.dispose();

  if (ui.modeLabel.textContent === "CNN entrenada") {
    return normalize(cnnScores);
  }

  const heuristicScores = scoreWithAcousticProfile(features.stats);
  return normalize(cnnScores.map((score, index) => score * 0.25 + heuristicScores[index] * 0.75));
}

function scoreWithAcousticProfile(stats) {
  const pitch = stats.pitch || 155;
  const centroid = stats.centroid || 1800;
  const bright = clamp((centroid - 1000) / 2600, 0, 1);
  const high = clamp((pitch - 190) / 170, 0, 1);
  const low = clamp((150 - pitch) / 95, 0, 1);
  const mid = 1 - Math.abs(clamp((pitch - 155) / 120, -1, 1));

  return [
    0.15 + high * 0.55 + bright * 0.3,
    0.2 + mid * 0.3 + bright * 0.35,
    0.2 + mid * 0.35 + (1 - bright) * 0.35,
    0.15 + low * 0.6 + (1 - bright) * 0.25,
  ];
}

function extractFeatures(samples, sampleRate) {
  const trimmed = trimSilence(samples);
  const window = hannWindow(FFT_SIZE);
  const filters = createMelFilterBank(MEL_BINS, FFT_SIZE, sampleRate);
  const frames = [];
  const frameCount = Math.max(1, Math.floor((trimmed.length - FFT_SIZE) / HOP_SIZE));
  const stride = Math.max(1, Math.floor(frameCount / TIME_STEPS));

  for (let frame = 0; frame < TIME_STEPS; frame++) {
    const start = Math.min(frame * stride * HOP_SIZE, Math.max(0, trimmed.length - FFT_SIZE));
    const slice = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) {
      slice[i] = (trimmed[start + i] || 0) * window[i];
    }
    const spectrum = magnitudeSpectrum(slice);
    frames.push(applyMelFilters(spectrum, filters));
  }

  const logMel = flattenAndNormalize(frames);
  return {
    melSpectrogram: logMel,
    stats: calculateStats(trimmed, sampleRate),
  };
}

function mixToMono(buffer) {
  const output = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < data.length; i++) {
      output[i] += data[i] / buffer.numberOfChannels;
    }
  }
  return output;
}

function trimSilence(samples) {
  const threshold = 0.015;
  let start = 0;
  let end = samples.length - 1;
  while (start < samples.length && Math.abs(samples[start]) < threshold) start++;
  while (end > start && Math.abs(samples[end]) < threshold) end--;
  if (start >= samples.length) return samples.slice(0, Math.min(samples.length, sampleFallbackLength()));
  return samples.slice(Math.max(0, start - 2048), Math.min(samples.length, end + 2048));
}

function sampleFallbackLength() {
  return FFT_SIZE * 2;
}

function magnitudeSpectrum(frame) {
  const spectrum = new Float32Array(FFT_SIZE / 2 + 1);
  for (let k = 0; k < spectrum.length; k++) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < FFT_SIZE; n++) {
      const angle = (2 * Math.PI * k * n) / FFT_SIZE;
      real += frame[n] * Math.cos(angle);
      imag -= frame[n] * Math.sin(angle);
    }
    spectrum[k] = Math.sqrt(real * real + imag * imag);
  }
  return spectrum;
}

function applyMelFilters(spectrum, filters) {
  const result = new Float32Array(MEL_BINS);
  for (let m = 0; m < MEL_BINS; m++) {
    let energy = 0;
    for (let k = 0; k < spectrum.length; k++) {
      energy += spectrum[k] * filters[m][k];
    }
    result[m] = Math.log10(energy + 1e-6);
  }
  return result;
}

function flattenAndNormalize(frames) {
  let min = Infinity;
  let max = -Infinity;
  for (const frame of frames) {
    for (const value of frame) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }

  const range = max - min || 1;
  const output = new Float32Array(MEL_BINS * TIME_STEPS);
  for (let t = 0; t < TIME_STEPS; t++) {
    for (let m = 0; m < MEL_BINS; m++) {
      output[m * TIME_STEPS + t] = (frames[t][m] - min) / range;
    }
  }
  return output;
}

function calculateStats(samples, sampleRate) {
  if (!samples.length) {
    return { energy: 0, pitch: 0, centroid: 0 };
  }

  const rms = Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
  const pitch = estimatePitch(samples, sampleRate);
  const spectrum = magnitudeSpectrum(samples.slice(0, FFT_SIZE));
  let weighted = 0;
  let total = 0;
  for (let i = 1; i < spectrum.length; i++) {
    const frequency = (i * sampleRate) / FFT_SIZE;
    weighted += frequency * spectrum[i];
    total += spectrum[i];
  }
  return {
    energy: rms,
    pitch,
    centroid: total ? weighted / total : 0,
  };
}

function estimatePitch(samples, sampleRate) {
  const maxSamples = Math.min(samples.length, sampleRate * 2);
  const segment = samples.slice(0, maxSamples);
  const minLag = Math.floor(sampleRate / 500);
  const maxLag = Math.floor(sampleRate / 70);
  let bestLag = 0;
  let bestScore = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let score = 0;
    for (let i = 0; i < segment.length - lag; i += 2) {
      score += segment[i] * segment[i + lag];
    }
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }

  return bestLag ? sampleRate / bestLag : 0;
}

function createMelFilterBank(count, fftSize, sampleRate) {
  const minMel = hzToMel(70);
  const maxMel = hzToMel(Math.min(8000, sampleRate / 2));
  const melPoints = Array.from({ length: count + 2 }, (_, index) =>
    minMel + (index / (count + 1)) * (maxMel - minMel),
  );
  const hzPoints = melPoints.map(melToHz);
  const binPoints = hzPoints.map((hz) => Math.floor(((fftSize + 1) * hz) / sampleRate));
  const filters = [];

  for (let m = 1; m <= count; m++) {
    const filter = new Float32Array(fftSize / 2 + 1);
    const left = binPoints[m - 1];
    const center = binPoints[m];
    const right = binPoints[m + 1];

    for (let k = left; k < center; k++) {
      filter[k] = (k - left) / Math.max(1, center - left);
    }
    for (let k = center; k < right; k++) {
      filter[k] = (right - k) / Math.max(1, right - center);
    }
    filters.push(filter);
  }

  return filters;
}

function hannWindow(size) {
  return Float32Array.from({ length: size }, (_, index) => 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (size - 1)));
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function normalize(values) {
  const sum = values.reduce((total, value) => total + Math.max(0, value), 0) || 1;
  return values.map((value) => Math.max(0, value) / sum);
}

function renderResult(scores, stats) {
  const bestIndex = scores.indexOf(Math.max(...scores));
  const confidence = scores[bestIndex];
  ui.primaryLabel.textContent = LABELS[bestIndex].name;
  ui.recommendation.textContent = LABELS[bestIndex].advice;
  ui.confidenceLabel.textContent = `${Math.round(confidence * 100)}%`;
  ui.confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
  ui.pitchValue.textContent = stats.pitch ? `${Math.round(stats.pitch)} Hz` : "--";
  ui.centroidValue.textContent = stats.centroid ? `${Math.round(stats.centroid)} Hz` : "--";
  ui.energyValue.textContent = `${Math.round(stats.energy * 1000) / 1000}`;
  renderProbabilities(scores);
}

function renderProbabilities(scores) {
  ui.probabilities.innerHTML = LABELS.map((label, index) => {
    const percent = Math.round((scores[index] || 0) * 100);
    return `
      <article class="probability">
        <strong>${label.name}</strong>
        <div class="mini-bar"><i style="width: ${percent}%"></i></div>
        <span>${percent}%</span>
      </article>
    `;
  }).join("");
}

function drawEmptySpectrogram() {
  const canvas = ui.spectrogram;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0b0e10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#6e7e83";
  ctx.font = "24px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("Carga o graba audio", canvas.width / 2, canvas.height / 2);
}

function drawSpectrogram(data) {
  const canvas = ui.spectrogram;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(TIME_STEPS, MEL_BINS);

  for (let m = 0; m < MEL_BINS; m++) {
    for (let t = 0; t < TIME_STEPS; t++) {
      const value = data[m * TIME_STEPS + t];
      const color = spectrogramColor(value);
      const offset = (m * TIME_STEPS + t) * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
  }

  const temp = document.createElement("canvas");
  temp.width = TIME_STEPS;
  temp.height = MEL_BINS;
  temp.getContext("2d").putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(temp, 0, 0, canvas.width, canvas.height);
}

function spectrogramColor(value) {
  const v = clamp(value, 0, 1);
  return [
    Math.round(18 + v * 232),
    Math.round(28 + Math.sin(v * Math.PI) * 190),
    Math.round(36 + (1 - v) * 122),
  ];
}

function setBusy(isBusy) {
  ui.analyzeButton.disabled = isBusy || !audioBuffer;
  ui.analyzeButton.textContent = isBusy ? "Analizando..." : "Analizar";
}

function formatSeconds(value) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

document.addEventListener("DOMContentLoaded", initializeUI);
