"use strict";

// Prompt labels and text from NVIDIA-AI-IOT/live-vlm-webui (Apache-2.0).
// Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
// Source: 2fd5ba0b334c334d24bf0f9439d8742b243d22be, static/index.html#promptPreset.
const PROMPT_PRESETS = [
  {
    "prompt": "Describe what you see in this image in one sentence.",
    "label": "Scene Description"
  },
  {
    "prompt": "List all objects you can see in this image, separated by commas.",
    "label": "Object Detection"
  },
  {
    "prompt": "Describe the person's activity and what they are doing.",
    "label": "Activity Recognition"
  },
  {
    "prompt": "Are there any safety hazards visible? Answer with 'ALERT: description' or 'SAFE'.",
    "label": "Safety Monitoring"
  },
  {
    "prompt": "Describe the facial expressions and emotions of people visible.",
    "label": "Emotion Detection"
  },
  {
    "prompt": "Provide a detailed description of the scene for a visually impaired person.",
    "label": "Accessibility"
  },
  {
    "prompt": "Read and transcribe any text visible in the image.",
    "label": "OCR / Text Reading"
  },
  {
    "prompt": "Answer with Yes or No only: Is there a person visible?",
    "label": "Yes/No Question"
  },
  {
    "prompt": "You are a robot. Describe what you see, then give 5 navigation commands to get to a possible location of a bathroom. Format: 'linear_x=0.3, angular_z=0.0 # reason'. Keep linear_x between -0.5 and 0.5, angular_z between -1.0 and 1.0.",
    "label": "🤖 Robot Navigation (Simple)"
  },
  {
    "prompt": "You are controlling a mobile robot with differential drive. Camera: 1.0m height, forward-facing. Generate ROS cmd_vel commands based on what you see to get to a possible location of a bathroom.\n\nCRITICAL CONSTRAINTS:\n- linear.x: MUST be between -0.5 and +0.5 m/s (violations will cause robot damage)\n- angular.z: MUST be between -1.0 and +1.0 rad/s\n- Commands must respond to actual obstacles/scene content\n\nOUTPUT FORMAT (20 commands for 2.0 seconds):\nT=0.0s: linear.x=0.20, angular.z=0.00 # [explain what you see and why this velocity]\nT=0.1s: linear.x=0.18, angular.z=-0.15 # [explain decision based on scene]\n... (continue for all 20 timesteps)\n\nSTRATEGY SECTION:\nAfter commands, explain: obstacles detected, clearance margins, goal inference, safety considerations.",
    "label": "🤖 Robot Navigation (ROS)"
  }
];

const CAPTURE_PRESETS = {
  "live-vlm": {cameraConstraints: {width: {ideal: 1280}, height: {ideal: 720}},
    temperature: 0.7, jpegQuality: 0.75, maxSide: null},
  lightweight: {cameraConstraints: {width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 15, max: 30}},
    temperature: 0, jpegQuality: 0.8, maxSide: 512}
};

// Count video frames, not wall-clock seconds. Delayed browser callbacks can
// cross several sampling boundaries; only the latest frame is eligible.
class FrameCadence {
  constructor(every = 30) { this.every = every; this.previous = null; this.count = 0; }
  observe(presentedFrames) {
    if (!Number.isInteger(presentedFrames) || presentedFrames < 1) return false;
    if (this.previous === null || presentedFrames < this.previous) {
      this.previous = presentedFrames; this.count = 1; return this.every === 1;
    }
    const before = Math.floor(this.count / this.every);
    this.count += presentedFrames - this.previous;
    this.previous = presentedFrames;
    return Math.floor(this.count / this.every) > before;
  }
}

const ADVANCED_DEFAULTS = {imageTokens: 512, imageTokenLimit: 512, topP: 1};

function validateAdvancedSettings(imageTokens, topP, limit = ADVANCED_DEFAULTS.imageTokenLimit) {
  if (!Number.isInteger(imageTokens) || imageTokens < 4 || imageTokens > limit) {
    throw new Error(`Choose an input image token budget from 4 to ${limit}, in whole tokens.`);
  }
  if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new Error("Sampling top-p must be greater than 0 and at most 1.");
  }
  return {imageTokens, topP};
}

// Only the native model boundary is comparable. Browser clocks and broader
// server request durations include JPEG work or transport and never substitute.
function readServerInferenceMs(metrics) {
  if (!metrics || metrics.timing_boundary !== "native_inference" ||
      metrics.timing_source !== "server_monotonic" ||
      typeof metrics.request_id !== "string" || !metrics.request_id.trim() ||
      !Number.isInteger(metrics.completion_tokens) || metrics.completion_tokens < 0 ||
      typeof metrics.native_inference_ms !== "number" ||
      !Number.isFinite(metrics.native_inference_ms) || metrics.native_inference_ms < 0) return null;
  return metrics.native_inference_ms;
}

// Server observation of the first nonempty text delta, before transport. Its
// consumer-scheduling delay can put it after native generation has completed.
function readServerFirstTextMs(metrics) {
  if (!metrics || metrics.timing_boundary !== "native_inference" ||
      metrics.first_text_timing_boundary !== "native_start_to_server_text" ||
      metrics.timing_source !== "server_monotonic" ||
      typeof metrics.request_id !== "string" || !metrics.request_id.trim() ||
      !Number.isInteger(metrics.completion_tokens) || metrics.completion_tokens < 0 ||
      typeof metrics.server_first_text_ms !== "number" ||
      !Number.isFinite(metrics.server_first_text_ms) || metrics.server_first_text_ms < 0) return null;
  return metrics.server_first_text_ms;
}

// Last server inference duration and the unrounded arithmetic mean over
// successful timed requests in one settings group; never a browser duration.
class LatencySummary {
  constructor() { this.reset(); }
  reset() { this.count = 0; this.totalMs = 0; this.lastMs = null; }
  add(milliseconds) {
    if (!Number.isFinite(milliseconds) || milliseconds < 0) return false;
    this.count += 1; this.totalMs += milliseconds; this.lastMs = milliseconds;
    return true;
  }
  get averageMs() { return this.count ? this.totalMs / this.count : null; }
}

// Handles arbitrary UTF-8/network chunk boundaries, LF/CRLF/CR, comments and
// multiple data lines. The caller supplies a streaming TextDecoder.
class SSEParser {
  constructor(onEvent) { this.onEvent = onEvent; this.buffer = ""; this.data = []; this.event = "message"; }
  feed(text, final = false) {
    this.buffer += text;
    if (this.buffer.length > 1048576) throw new Error("SSE event exceeded the size limit.");
    while (true) {
      const match = /[\r\n]/.exec(this.buffer);
      if (!match) break;
      const position = match.index;
      if (this.buffer[position] === "\r" && position === this.buffer.length - 1 && !final) break;
      const size = this.buffer[position] === "\r" && this.buffer[position + 1] === "\n" ? 2 : 1;
      const line = this.buffer.slice(0, position);
      this.buffer = this.buffer.slice(position + size);
      this.line(line);
    }
    // EOF does not dispatch an event: a real blank-line boundary is required.
    // In particular, a truncated final [DONE] must not count as completion.
    if (final) { this.buffer = ""; this.data = []; this.event = "message"; }
  }
  line(line) {
    if (!line) {
      if (this.data.length) this.onEvent({type: this.event, data: this.data.join("\n")});
      this.data = []; this.event = "message";
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
    if (field === "event") this.event = value;
  }
}

function readCompletionEvent(event) {
  if (event.data === "[DONE]") return {done: true};
  const data = JSON.parse(event.data);
  if (data.error) throw new Error(data.error.message || "Backend stream failed.");
  const choice = data.choices?.[0];
  const finishReason = choice?.finish_reason;
  if (["error", "cancelled", "canceled"].includes(finishReason)) {
    throw new Error(`Backend ended generation with status: ${finishReason}.`);
  }
  return {text: choice?.delta?.content, finishReason,
    ...(Object.hasOwn(data, "cosmos_metrics") ? {metrics: data.cosmos_metrics} : {})};
}

// Original history/drawing code; the colored sparklines are inspired by Live VLM WebUI.
// Points use device sample timestamps. Missing values remain gaps; genuine 0% remains data.
function appendTelemetrySample(history, sample) {
  if (!Number.isFinite(sample?.at)) return history.slice();
  const last = history[history.length - 1];
  if (last && sample.at === last.at) return history.slice();
  const backwards = last && sample.at < last.at;
  const retained = backwards ? [] : history.filter(point => point.at >= sample.at - 60000);
  const valid = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100 ? value : null;
  return [...retained.slice(-119), {at: sample.at, cpu: valid(sample.cpu), gpu: valid(sample.gpu),
    memory: valid(sample.memory), breakBefore: Boolean(sample.breakBefore || backwards)}];
}

function telemetrySegments(history, key, windowEnd) {
  const segments = [];
  let current = null, previous = null;
  for (const point of history) {
    if (point.at < windowEnd - 60000 || point.at > windowEnd) continue;
    const value = point[key];
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 100) {
      current = null; previous = null;
      continue;
    }
    // The sampler runs once a second. A longer interval does not imply observed continuity.
    if (!current || point.breakBefore || point.at - previous.at > 1500 || point.at <= previous.at) {
      current = []; segments.push(current);
    }
    current.push({at: point.at, value}); previous = point;
  }
  return segments;
}

// Device telemetry is independent of inference and camera request ownership.
// A failed or stale sample clears the values instead of leaving them looking live.
function startDeviceTelemetry() {
  const nodes = Object.fromEntries(["cpuUsage", "gpuUsage", "memoryAmount", "memoryPercent",
    "cpuMeter", "gpuMeter", "memoryMeter", "telemetryStatus"].map(id => [id, document.getElementById(id)]));
  if (Object.values(nodes).some(node => !node)) return;
  let active = null, interval = null, lastSample = null;
  let history = [], clockAnchor = null, gapPending = true;
  const charts = [
    {key: "cpu", name: "CPU usage", canvas: document.getElementById("cpuHistory"), scale: document.getElementById("cpuHistoryScale")},
    {key: "gpu", name: "GPU usage", canvas: document.getElementById("gpuHistory"), scale: document.getElementById("gpuHistoryScale")},
    {key: "memory", name: "System shared RAM usage", canvas: document.getElementById("memoryHistory"), scale: document.getElementById("memoryHistoryScale")},
  ];
  const percent = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100 ? value : null;
  const bytes = value => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  function drawHistories() {
    const end = clockAnchor ? clockAnchor.at + performance.now() - clockAnchor.receivedAt : 0;
    history = history.filter(point => point.at >= end - 60000);
    for (const {key, name, canvas, scale} of charts) {
      if (!canvas || typeof canvas.getContext !== "function") continue;
      const context = canvas.getContext("2d");
      if (!context) continue;
      const {width, height} = canvas.getBoundingClientRect();
      if (!width || !height) continue;
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth; canvas.height = pixelHeight;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      const segments = telemetrySegments(history, key, end);
      const values = segments.flat().map(point => point.value);
      const maximum = Math.max(1, ...values);
      if (scale) scale.textContent = values.length ? `0–${maximum.toFixed(1)}%` : "No samples";
      const color = getComputedStyle(canvas).color;
      const bottom = height - 3;
      const x = point => 3 + (point.at - (end - 60000)) / 60000 * (width - 6);
      const y = point => bottom - point.value / maximum * (height - 6);
      context.strokeStyle = color; context.fillStyle = color;
      context.lineWidth = 1.6; context.lineJoin = "round"; context.lineCap = "round";
      for (const segment of segments) {
        context.beginPath(); context.moveTo(x(segment[0]), bottom);
        for (const point of segment) context.lineTo(x(point), y(point));
        context.lineTo(x(segment[segment.length - 1]), bottom); context.closePath();
        context.globalAlpha = 0.14; context.fill(); context.globalAlpha = 1;
        context.beginPath(); context.moveTo(x(segment[0]), y(segment[0]));
        for (const point of segment.slice(1)) context.lineTo(x(point), y(point));
        context.stroke();
        if (segment.length === 1) {
          context.beginPath(); context.arc(x(segment[0]), y(segment[0]), 1.8, 0, Math.PI * 2); context.fill();
        }
      }
      canvas.setAttribute("aria-label", `${name}, last 60 seconds. ${values.length
        ? `Chart scale 0 to ${maximum.toFixed(1)} percent. Observed range ${Math.min(...values).toFixed(1)} to ${Math.max(...values).toFixed(1)} percent. Latest plotted value ${values[values.length - 1].toFixed(1)} percent.`
        : "No available samples in this period."}`);
    }
  }
  function status(message, kind) {
    nodes.telemetryStatus.textContent = message;
    nodes.telemetryStatus.className = `telemetry-status ${kind}`;
  }
  function clear(message, kind = "unavailable") {
    lastSample = null;
    gapPending = true;
    nodes.cpuUsage.textContent = "—"; nodes.gpuUsage.textContent = "—";
    nodes.memoryAmount.textContent = "—"; nodes.memoryPercent.textContent = "Unavailable";
    for (const name of ["cpuMeter", "gpuMeter", "memoryMeter"]) nodes[name].style.width = "0%";
    status(message, kind);
    drawHistories();
  }
  async function poll() {
    if (active || document.visibilityState !== "visible") return;
    const controller = new AbortController();
    active = controller;
    const started = performance.now();
    const timeout = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch("/api/metrics", {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw new Error("Metrics unavailable");
      const sample = await response.json();
      if (controller.signal.aborted || active !== controller || document.visibilityState !== "visible") return;
      if (!sample || !["ok", "partial"].includes(sample.status) ||
          !Number.isFinite(Date.parse(sample.sampled_at)) ||
          typeof sample.age_ms !== "number" || !Number.isFinite(sample.age_ms) || sample.age_ms < 0) {
        throw new Error("Invalid or unavailable metrics");
      }
      // Server age avoids relying on synchronized browser and device clocks.
      // Including request duration is a conservative bound on transit age.
      const age = sample.age_ms + performance.now() - started;
      if (age > 5000) { clear("Stale · waiting for fresh metrics", "stale"); return; }
      const cpu = percent(sample.cpu?.utilization_percent);
      const gpu = percent(sample.gpu?.utilization_percent);
      const memory = sample.memory;
      const used = bytes(memory?.used_bytes), total = bytes(memory?.total_bytes);
      const validMemory = used !== null && total !== null && total > 0 && used <= total &&
        memory.shared === true && memory.measurement === "MemTotal - MemAvailable";
      const memoryPercent = validMemory ? percent(memory.utilization_percent) : null;
      nodes.cpuUsage.textContent = cpu === null ? "—" : `${cpu.toFixed(1)}%`;
      nodes.gpuUsage.textContent = gpu === null ? "—" : `${gpu.toFixed(1)}%`;
      nodes.memoryAmount.textContent = validMemory ? `${(used / 2**30).toFixed(2)} / ${(total / 2**30).toFixed(2)} GiB` : "—";
      nodes.memoryPercent.textContent = memoryPercent === null ? "Unavailable" : `${memoryPercent.toFixed(1)}%`;
      nodes.cpuMeter.style.width = `${cpu ?? 0}%`;
      nodes.gpuMeter.style.width = `${gpu ?? 0}%`;
      nodes.memoryMeter.style.width = `${memoryPercent ?? 0}%`;
      if (cpu === null && gpu === null && !validMemory) { clear("Device metrics unavailable"); return; }
      const sampledAt = Date.parse(sample.sampled_at);
      const previousAt = history[history.length - 1]?.at;
      history = appendTelemetrySample(history, {at: sampledAt, cpu, gpu, memory: memoryPercent, breakBefore: gapPending});
      if (sampledAt !== previousAt) gapPending = false;
      clockAnchor = {at: sampledAt + age, receivedAt: performance.now()};
      drawHistories();
      lastSample = {age, receivedAt: performance.now()};
      const partial = sample.status === "partial" || cpu === null || gpu === null || memoryPercent === null;
      status(partial ? "Partial metrics · some readings unavailable" : "Live · updates every second", partial ? "partial" : "live");
    } catch (_) {
      if (active === controller && document.visibilityState === "visible") clear("Device metrics unavailable");
    } finally {
      clearTimeout(timeout);
      if (active === controller) active = null;
    }
  }
  function tick() {
    if (lastSample && lastSample.age + performance.now() - lastSample.receivedAt > 5000) {
      clear("Stale · waiting for fresh metrics", "stale");
    }
    drawHistories();
    void poll();
  }
  function pause() {
    clearInterval(interval); interval = null;
    active?.abort();
    clear("Paused · page not visible", "paused");
  }
  function resume() {
    clearInterval(interval); interval = null;
    if (document.visibilityState !== "visible") { pause(); return; }
    clear("Refreshing device metrics…");
    tick(); interval = setInterval(tick, 1000);
  }
  document.addEventListener("visibilitychange", resume);
  window.addEventListener("pagehide", pause);
  window.addEventListener("pageshow", resume);
  window.addEventListener("resize", drawHistories);
  if (typeof ResizeObserver !== "undefined") {
    const resizeObserver = new ResizeObserver(drawHistories);
    for (const {canvas} of charts) if (canvas) resizeObserver.observe(canvas);
  }
  resume();
}

if (typeof module !== "undefined") module.exports = {SSEParser, readCompletionEvent, CAPTURE_PRESETS, FrameCadence, PROMPT_PRESETS, LatencySummary, appendTelemetrySample, telemetrySegments, ADVANCED_DEFAULTS, validateAdvancedSettings, readServerInferenceMs, readServerFirstTextMs};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const state = {ready: false, model: "", running: false, busy: false, media: null,
    abort: null, captureAt: null, completed: 0, imageURL: null, checking: false, cameraGeneration: 0,
    preset: "lightweight", frameCallback: null, sampled: 0, skipped: 0,
    liveStreaming: true, activeTrigger: null, engineId: null, timingGroup: 0,
    advanced: {...ADVANCED_DEFAULTS}, advancedValid: true};
  const canvas = document.createElement("canvas");
  const duration = ms => ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
  const latency = new LatencySummary();
  let serverFirstTextMs = null;
  function renderLatency() {
    $("serverTtftValue").textContent = serverFirstTextMs === null ? "—" : String(Math.round(serverFirstTextMs));
    $("latencyValue").textContent = latency.lastMs === null ? "—" : String(Math.round(latency.lastMs));
    $("avgLatencyValue").textContent = latency.averageMs === null ? "—" : String(Math.round(latency.averageMs));
    $("countValue").textContent = String(latency.count);
  }
  function resetTimingGroup(message) {
    state.timingGroup += 1;
    serverFirstTextMs = null; latency.reset(); renderLatency();
    $("timingStatus").textContent = message;
  }
  function advancedValues() {
    const selected = $("imageTokenPreset").value;
    return validateAdvancedSettings(Number(selected === "custom" ? $("customImageTokens").value : selected),
      Number($("topP").value), state.advanced.imageTokenLimit);
  }
  function applyAdvancedControls(reset = true) {
    $("customImageTokensLabel").hidden = $("imageTokenPreset").value !== "custom";
    try {
      const values = advancedValues();
      const changed = values.imageTokens !== state.advanced.imageTokens || values.topP !== state.advanced.topP;
      Object.assign(state.advanced, values);
      state.advancedValid = true;
      $("advancedError").hidden = true; $("advancedError").textContent = "";
      if (changed && reset) resetTimingGroup("Settings changed · waiting for a timed answer with these settings.");
    } catch (err) {
      state.advancedValid = false;
      $("advancedError").hidden = false; $("advancedError").textContent = err.message;
    }
    controls();
  }
  function applyRuntime(runtime) {
    if (!runtime || typeof runtime.engine_id !== "string" || !runtime.engine_id.trim() ||
        !Number.isInteger(runtime.max_image_tokens_per_image_limit) || runtime.max_image_tokens_per_image_limit < 4 ||
        typeof runtime.static_clocks !== "boolean" || !Number.isInteger(runtime.encoder_cache_bytes) ||
        runtime.encoder_cache_bytes < 0) throw new Error("Invalid engine settings");
    const limit = Math.min(512, runtime.max_image_tokens_per_image_limit);
    const defaults = validateAdvancedSettings(runtime.max_image_tokens_per_image, runtime.top_p, limit);
    const engineChanged = state.engineId !== runtime.engine_id;
    const capacityChanged = state.advanced.imageTokenLimit !== limit;
    state.advanced.imageTokenLimit = limit;
    $("customImageTokens").max = String(limit);
    $("imageTokenCapacity").textContent = `Loaded engine supports 4–${limit} input image tokens. This does not change the output token limit.`;
    for (const option of $("imageTokenPreset").options || []) {
      if (option.value !== "custom") option.disabled = Number(option.value) > limit;
    }
    if (engineChanged) {
      state.engineId = runtime.engine_id;
      $("imageTokenPreset").value = [320, 512].includes(defaults.imageTokens) ? String(defaults.imageTokens) : "custom";
      $("customImageTokens").value = String(defaults.imageTokens);
      $("topP").value = String(defaults.topP);
      applyAdvancedControls(false);
      resetTimingGroup("Engine loaded · waiting for a timed answer with its defaults.");
    } else if (capacityChanged) {
      applyAdvancedControls(false);
      resetTimingGroup("Engine capacity changed · waiting for a timed answer.");
    }
    $("staticClocksValue").textContent = runtime.static_clocks ? "Enabled (1)" : "Disabled (0)";
    $("encoderCacheValue").textContent = `${(runtime.encoder_cache_bytes / 2**20).toLocaleString(undefined, {maximumFractionDigits: 2})} MiB`;
    $("runtimeStatus").textContent = "Active engine settings reported by the server. Fixed clocks and encoder cache are read-only here; changing them requires a backend restart.";
  }
  $("imageTokenPreset").value = String(ADVANCED_DEFAULTS.imageTokens);
  $("customImageTokens").value = String(ADVANCED_DEFAULTS.imageTokens);
  $("topP").value = String(ADVANCED_DEFAULTS.topP);
  $("imageTokenPreset").addEventListener("change", () => applyAdvancedControls());
  $("customImageTokens").addEventListener("input", () => applyAdvancedControls());
  $("topP").addEventListener("input", () => applyAdvancedControls());
  // Move one caption element between docks, keeping both the live video and
  // active token stream intact. No duplicated or separately updated captions.
  const captionPositions = new Set(["side", "above", "below"]);
  const captionStorageKey = "cosmos3-edge:caption-position";
  function applyCaptionPosition(position, save = true) {
    if (!captionPositions.has(position)) position = "below";
    $("workspace").dataset.captionPosition = position;
    const docks = {above: "captionAbove", below: "captionBelow", side: "captionSide"};
    for (const [name, id] of Object.entries(docks)) $(id).hidden = name !== position;
    $(docks[position]).append($("answer"));
    for (const input of document.querySelectorAll('input[name="captionPosition"]')) {
      input.checked = input.value === position;
    }
    if (save) { try { localStorage.setItem(captionStorageKey, position); } catch (_) {} }
  }
  let savedCaptionPosition = "below";
  try { savedCaptionPosition = localStorage.getItem(captionStorageKey) || "below"; } catch (_) {}
  applyCaptionPosition(savedCaptionPosition, false);
  for (const input of document.querySelectorAll('input[name="captionPosition"]')) {
    input.addEventListener("change", () => { if (input.checked) applyCaptionPosition(input.value); });
  }

  const customPrompt = document.createElement("option");
  customPrompt.value = ""; customPrompt.textContent = "Custom prompt / select a preset";
  $("promptPreset").append(customPrompt);
  for (const preset of PROMPT_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.prompt; option.textContent = preset.label;
    $("promptPreset").append(option);
  }
  function matchPromptPreset() {
    $("promptPreset").value = PROMPT_PRESETS.some(preset => preset.prompt === $("prompt").value)
      ? $("prompt").value : "";
  }
  $("promptPreset").addEventListener("change", () => {
    if ($("promptPreset").value) $("prompt").value = $("promptPreset").value;
  });
  $("prompt").addEventListener("input", matchPromptPreset);
  matchPromptPreset();
  function error(message = "") { $("error").textContent = message; $("error").hidden = !message; }
  function controls() {
    $("startButton").disabled = state.running || state.busy;
    $("stopButton").disabled = !state.running && !state.busy;
    const sourceReady = state.running ? Boolean(state.media && $("video").readyState >= 2) : Boolean(state.imageURL);
    $("analyzeButton").disabled = !sourceReady || !state.ready || state.busy || !state.advancedValid;
    $("liveToggleButton").setAttribute("aria-pressed", String(state.liveStreaming));
    $("liveToggleButton").textContent = `Live streaming: ${state.liveStreaming ? "On" : "Off"}`;
  }
  async function checkBackend() {
    if (state.checking) return;
    state.checking = true;
    try {
      const [health, models, runtime] = await Promise.all([
        fetch("/health/ready", {signal: AbortSignal.timeout(5000)}),
        fetch("/v1/models", {signal: AbortSignal.timeout(5000)}),
        fetch("/api/runtime", {cache: "no-store", signal: AbortSignal.timeout(5000)})
          .then(response => response.ok ? response.json() : null).catch(() => null)
      ]);
      if (!health.ok || !models.ok) throw new Error("unavailable");
      const healthData = await health.json();
      const modelsData = await models.json();
      const model = modelsData.data?.[0]?.id;
      if (healthData.status !== "ready" || !model) throw new Error("not ready");
      state.ready = true; state.model = model;
      $("backendStatus").textContent = "Local backend ready";
      $("backendStatus").className = "badge ready";
      $("modelName").textContent = model;
      try { applyRuntime(runtime); } catch (_) {
        $("staticClocksValue").textContent = "Unavailable";
        $("encoderCacheValue").textContent = "Unavailable";
        $("runtimeStatus").textContent = "Engine settings unavailable. Input controls keep their current values; clocks and cache cannot be verified. Load defaults are shown above.";
      }
    } catch (_) {
      state.ready = false;
      $("backendStatus").textContent = "Backend not ready";
      $("backendStatus").className = "badge unavailable";
      $("modelName").textContent = "Waiting for local TensorRT-Edge-LLM";
      $("staticClocksValue").textContent = "Unavailable"; $("encoderCacheValue").textContent = "Unavailable";
      $("runtimeStatus").textContent = "Waiting for the backend to report active engine settings. Load defaults are shown above.";
      if (!state.busy) $("runStatus").textContent = "Waiting for local backend";
    } finally { state.checking = false; controls(); }
  }
  function capture(source) {
    const width = source.videoWidth || source.naturalWidth;
    const height = source.videoHeight || source.naturalHeight;
    if (!width || !height) throw new Error("The image is not ready yet.");
    const preset = CAPTURE_PRESETS[state.preset];
    const limit = state.preset === "lightweight" ? Number($("resolution").value) : null;
    const ratio = limit === null ? 1 : Math.min(1, limit / Math.max(width, height));
    const imageWidth = Math.max(1, Math.round(width * ratio));
    const imageHeight = Math.max(1, Math.round(height * ratio));
    // Preserve the whole image without upscaling while keeping both canvas
    // axes in the supported visual position grid (at least 16 patches).
    canvas.width = Math.max(256, imageWidth);
    canvas.height = Math.max(256, imageHeight);
    if (state.preset === "live-vlm" && (canvas.width > 4096 || canvas.height > 4096 ||
        Math.max(canvas.width, canvas.height) / Math.min(canvas.width, canvas.height) > 8)) {
      throw new Error("This full-size image is too large or too wide for the selected model profile. Choose a source up to 4096 pixels per side and an aspect ratio within 8:1, or use Lightweight.");
    }
    const context = canvas.getContext("2d", {alpha: false});
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, Math.floor((canvas.width - imageWidth) / 2),
      Math.floor((canvas.height - imageHeight) / 2), imageWidth, imageHeight);
    const capturedAt = performance.now();
    const url = canvas.toDataURL("image/jpeg", preset.jpegQuality);
    if (url.length > 1900000) throw new Error("This full-size frame exceeds the image request limit. Choose a smaller source or the Lightweight preset.");
    $("captureStatus").textContent = `Sent ${canvas.width}×${canvas.height} · ${state.preset === "live-vlm" ? "Live VLM WebUI" : "Lightweight"}`;
    return {url, capturedAt};
  }
  async function analyze(source, trigger = "manual") {
    if (state.busy || !state.ready || !state.advancedValid) return;
    const prompt = $("prompt").value.trim();
    const maxTokens = Number($("maxTokens").value);
    if (!prompt || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) {
      error("Enter a prompt and an output token limit from 1 to 512.");
      stop(); return;
    }
    // Freeze request settings and group ownership before capture. Later UI or
    // engine changes affect the next request and cannot mix timing populations.
    let advanced;
    try { advanced = advancedValues(); } catch (err) { error(err.message); return; }
    const timingGroup = state.timingGroup;
    const temperature = CAPTURE_PRESETS[state.preset].temperature;
    const controller = new AbortController();
    state.busy = true; state.abort = controller; state.activeTrigger = trigger; controls(); error();
    $("ttft").textContent = "—"; $("totalTime").textContent = "—";
    $("runStatus").textContent = "Reading frame…";
    let reader;
    try {
      const image = capture(source);
      state.captureAt = image.capturedAt;
      if (state.running && source === $("video")) state.sampled += 1;
      const started = performance.now();
      let firstToken = null, done = false, output = "", finishReason = null, serverMetrics = null;
      const response = await fetch("/v1/chat/completions", {
        method: "POST", headers: {"Content-Type": "application/json"}, signal: controller.signal,
        body: JSON.stringify({model: state.model, stream: true, temperature,
          top_p: advanced.topP, max_image_tokens_per_image: advanced.imageTokens,
          stream_options: {include_usage: true},
          max_tokens: maxTokens, messages: [{role: "user", content: [
            {type: "text", text: prompt}, {type: "image_url", image_url: {url: image.url}}
          ]}]})
      });
      controller.signal.throwIfAborted();
      if (state.abort !== controller) return;
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error?.message || `Backend returned HTTP ${response.status}.`);
      }
      if (!response.headers.get("content-type")?.includes("text/event-stream")) throw new Error("Backend did not return a token stream.");
      if (!response.body) throw new Error("Streaming responses are not supported in this browser.");
      const decoder = new TextDecoder();
      const parser = new SSEParser(event => {
        const completion = readCompletionEvent(event);
        if (completion.done) { done = true; return; }
        if (Object.hasOwn(completion, "metrics")) serverMetrics = completion.metrics;
        if (completion.finishReason) finishReason = completion.finishReason;
        const text = completion.text;
        if (typeof text === "string" && text.length) {
          if (firstToken === null) { firstToken = performance.now(); $("ttft").textContent = duration(firstToken - started); }
          output += text;
          if (output.length > 65536) throw new Error("Model output exceeded the display limit.");
          $("answer").textContent = output;
          $("answer").scrollTop = $("answer").scrollHeight;
          $("answer").classList.add("streaming");
          $("runStatus").textContent = "Writing answer…";
        }
      });
      reader = response.body.getReader();
      while (!done) {
        const chunk = await reader.read();
        controller.signal.throwIfAborted();
        if (state.abort !== controller) return;
        if (chunk.done) { parser.feed(decoder.decode(), true); break; }
        parser.feed(decoder.decode(chunk.value, {stream: true}));
      }
      if (!done) throw new Error("Connection ended before the model completed its answer.");
      if (!["stop", "length"].includes(finishReason)) throw new Error("Backend stream did not confirm a successful completion.");
      if (!output.trim()) throw new Error("The backend completed without visible answer text. Try a larger output token limit.");
      $("totalTime").textContent = duration(performance.now() - started);
      if (timingGroup === state.timingGroup) {
        serverFirstTextMs = readServerFirstTextMs(serverMetrics);
        const milliseconds = readServerInferenceMs(serverMetrics);
        if (milliseconds === null) {
          latency.lastMs = null;
          $("timingStatus").textContent = "Latest answer has no valid server inference timing; excluded from Average and Timed.";
        } else {
          latency.add(milliseconds);
          $("timingStatus").textContent = "Server-measured native inference · current settings only.";
        }
        renderLatency();
      }
      $("runStatus").textContent = finishReason === "length" ? "Output token limit reached" : "Answer complete";
      state.completed += 1; $("requestCount").textContent = `${state.completed} completed`;
    } catch (err) {
      if (state.abort !== controller) return;
      if (err.name === "AbortError" || controller.signal.aborted) $("runStatus").textContent = "Stopped · partial answer";
      else { error(err.message); $("runStatus").textContent = "Request failed"; state.running = false; }
    } finally {
      // Browser cancellation can remain pending after a fetch abort. It must
      // never hold the controls busy or clean up a later request's state.
      try { if (reader) reader.cancel().catch(() => {}); } catch (_) {}
      if (state.abort === controller) {
        state.abort = null; state.busy = false; state.activeTrigger = null;
        $("answer").classList.remove("streaming");
        if (!state.running) releaseCamera();
        controls();
      }
    }
  }
  async function cameraLoop(generation) {
    while (state.running && generation === state.cameraGeneration) {
      const start = performance.now();
      if (state.liveStreaming && state.ready && $("video").readyState >= 2) await analyze($("video"), "live");
      const delay = Math.max(100, Number($("interval").value) - (performance.now() - start));
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  function frameCameraLoop(generation) {
    const cadence = new FrameCadence(30);
    const video = $("video");
    function frame(now, metadata) {
      if (!state.running || generation !== state.cameraGeneration) return;
      state.frameCallback = null;
      if (cadence.observe(metadata.presentedFrames) && state.liveStreaming) {
        if (state.busy) state.skipped += 1;
        else if (state.ready && video.readyState >= 2) {
          void analyze(video, "live");
        }
        $("samplingStatus").textContent = `${state.sampled} frames sent · ${state.skipped} skipped while busy`;
      }
      if (state.running && generation === state.cameraGeneration) state.frameCallback = video.requestVideoFrameCallback(frame);
    }
    state.frameCallback = video.requestVideoFrameCallback(frame);
  }
  function releaseCamera() {
    if (state.frameCallback !== null) $("video").cancelVideoFrameCallback?.(state.frameCallback);
    state.frameCallback = null;
    if (state.media) state.media.getTracks().forEach(track => track.stop());
    state.media = null; $("video").srcObject = null; $("liveIndicator").hidden = true;
    $("sourceStatus").textContent = state.imageURL ? "Selected image" : "Camera stopped";
  }
  function stop() { state.running = false; state.cameraGeneration += 1; state.abort?.abort(); releaseCamera(); controls(); }
  function applyPreset(name, reset = true) {
    if (!CAPTURE_PRESETS[name]) return;
    if (reset) {
      stop();
      // Retire the aborted owner before clearing its result. Its late stream
      // cleanup must never overwrite the new preset or a restarted request.
      state.abort = null; state.busy = false; state.activeTrigger = null;
      state.completed = 0; state.captureAt = null;
      resetTimingGroup("Capture preset changed · waiting for a timed answer.");
      $("answer").textContent = "Preset changed. Start the camera or analyze your selected image.";
      $("answer").classList.remove("streaming");
      $("runStatus").textContent = "Waiting for input";
      $("requestCount").textContent = "0 completed";
      for (const id of ["ttft", "totalTime", "frameAge"]) $(id).textContent = "—";
      $("captureStatus").textContent = "No frame sent"; error();
    }
    state.preset = name; state.sampled = 0; state.skipped = 0;
    const live = name === "live-vlm";
    $("liveVlmPreset").checked = live; $("lightweightPreset").checked = !live;
    $("interval").value = live ? "frames30" : "1000";
    $("resolution").value = live ? "native" : "512";
    $("interval").disabled = live; $("resolution").disabled = live;
    $("presetDescription").textContent = live
      ? "Requests a 1280×720 camera with no FPS cap. Sends full-size frames every 30 video frames and skips samples while busy. Temperature 0.7."
      : "First demo defaults: requests a 640×480 camera at an ideal 15 FPS (maximum 30). Sends frames with a 512-pixel longest side, JPEG quality 0.8 and a 1-second minimum interval, adjustable below. Captures after each answer. Temperature 0.";
    $("samplingStatus").textContent = live ? "Every 30 frames · waiting for camera" : "Capture after each answer";
    controls();
  }
  $("liveVlmPreset").addEventListener("change", () => { if ($("liveVlmPreset").checked) applyPreset("live-vlm"); });
  $("lightweightPreset").addEventListener("change", () => { if ($("lightweightPreset").checked) applyPreset("lightweight"); });
  $("startButton").addEventListener("click", async () => {
    error();
    if (!navigator.mediaDevices?.getUserMedia) { error("Camera access needs HTTPS or http://localhost. You can choose an image instead."); return; }
    if (state.preset === "live-vlm" && !$("video").requestVideoFrameCallback) {
      error("This browser cannot count video frames. Use a current browser or choose Lightweight."); return;
    }
    state.running = true; const generation = ++state.cameraGeneration; controls();
    try {
      const media = await navigator.mediaDevices.getUserMedia({audio: false,
        video: CAPTURE_PRESETS[state.preset].cameraConstraints});
      if (!state.running || generation !== state.cameraGeneration) { media.getTracks().forEach(track => track.stop()); return; }
      state.media = media; $("video").srcObject = media; await $("video").play();
      $("video").hidden = false; $("uploadedImage").hidden = true; $("placeholder").hidden = true;
      $("liveIndicator").hidden = false;
      const actual = media.getVideoTracks()[0].getSettings();
      $("sourceStatus").textContent = `Camera · ${$("video").videoWidth}×${$("video").videoHeight}${actual.frameRate ? ` · ${actual.frameRate.toFixed(1)} FPS` : ""}`;
      state.sampled = 0; state.skipped = 0;
      $("samplingStatus").textContent = state.preset === "live-vlm" ? "Every 30 frames · waiting for first sample" : "Capture after each answer";
      if (!state.liveStreaming) $("samplingStatus").textContent = "Live streaming off · manual capture ready";
      controls();
      if (state.preset === "live-vlm") frameCameraLoop(generation); else cameraLoop(generation);
    } catch (err) { if (generation === state.cameraGeneration) { stop(); error(`Camera unavailable: ${err.message}. You can choose an image instead.`); } }
  });
  $("stopButton").addEventListener("click", stop);
  $("analyzeButton").addEventListener("click", () => analyze(state.running ? $("video") : $("uploadedImage")));
  $("liveToggleButton").addEventListener("click", () => {
    state.liveStreaming = !state.liveStreaming;
    // Pause automatic requests without stopping the preview or interrupting
    // an explicitly requested manual inference.
    if (!state.liveStreaming && state.activeTrigger === "live") state.abort?.abort();
    $("samplingStatus").textContent = state.liveStreaming
      ? (state.preset === "live-vlm" ? "Every 30 frames · live streaming on" : "Capture after each answer")
      : "Live streaming off · use Run inference";
    controls();
  });
  $("imageInput").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    stop(); error();
    if (!new Set(["image/jpeg", "image/png", "image/webp"]).has(file.type) || file.size > 12 * 1024 * 1024) { error("Choose a JPEG, PNG or WebP image smaller than 12 MiB."); return; }
    if (state.imageURL) URL.revokeObjectURL(state.imageURL);
    state.imageURL = URL.createObjectURL(file);
    $("uploadedImage").src = state.imageURL;
    try {
      await $("uploadedImage").decode();
      $("uploadedImage").hidden = false; $("video").hidden = true; $("placeholder").hidden = true;
      $("sourceStatus").textContent = "Selected image"; controls();
    } catch (_) { error("This image could not be decoded."); URL.revokeObjectURL(state.imageURL); state.imageURL = null; controls(); }
  });
  window.addEventListener("pagehide", stop);
  setInterval(() => { if (state.captureAt !== null) $("frameAge").textContent = duration(performance.now() - state.captureAt); }, 100);
  applyPreset("lightweight", false);
  setInterval(checkBackend, 5000); checkBackend(); controls();
  if (!window.isSecureContext) {
    $("cameraHelp").textContent = "Image upload works here. Camera access requires HTTPS.";
    fetch("/api/access", {signal: AbortSignal.timeout(3000)}).then(r => r.json()).then(access => {
      if (!Number.isInteger(access.https_port) || access.https_port < 1 || access.https_port > 65535) return;
      const url = new URL(window.location.href);
      url.protocol = "https:"; url.port = String(access.https_port); url.pathname = "/"; url.search = ""; url.hash = "";
      const link = document.createElement("a");
      link.href = url.href; link.textContent = "Open HTTPS for camera access";
      $("cameraHelp").append(" ", link, ". This device uses a local certificate.");
    }).catch(() => {});
  }
  startDeviceTelemetry();
}
