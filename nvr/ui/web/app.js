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

// Reachy Mini arrives as MJPEG through serve_ui.py's /reachy/ relay of the robot bridge,
// which publishes about 5 FPS. An <img> has no requestVideoFrameCallback, so the Live VLM
// WebUI "every 30 video frames" becomes the 6 s those 30 frames take, labelled as time-based.
const REACHY_BRIDGE_FPS = 5;
const REACHY_SAMPLE_MS = 30 / REACHY_BRIDGE_FPS * 1000;
const REACHY_HEALTH_MS = 2000;
// A health answer older than this no longer vouches for the stream.
const REACHY_HEALTH_STALE_MS = 6000;
// An MJPEG <img> fires load for its first frame only, and nothing at all when its stream ends
// cleanly (a restart of the relay, say) while it goes on showing the last frame. So each
// health poll also asks the relay about this page's stream. One it no longer has, or one that
// has relayed nothing for this long while the bridge is live, is reopened, and its picture is
// not sent for inference meanwhile. The relay is only asked about a stream open long enough to
// have reached it, and a new stream has the grace period to deliver its first frame.
const REACHY_STREAM_IDLE_MS = 3000;
const REACHY_STREAM_SETTLE_MS = 1000;
const REACHY_STREAM_GRACE_MS = 5000;

// X-Reachy-Stream from serve_ui.py: "closed", or "idle_ms=<n>" since it last relayed a byte.
// Anything else (an older server, a failed poll) is null: unknown, so no action is taken.
function readRelayStream(header) {
  if (header === "closed") return {open: false, idleMs: null};
  const match = /^idle_ms=(\d{1,9})$/.exec(header || "");
  return match ? {open: true, idleMs: Number(match[1])} : null;
}

// X-Reachy-Slots from serve_ui.py: "<open>/<max>" Reachy streams on the whole server, every
// tab's video and audio. A refused <img> cannot see its 503, so this is how the page tells a
// full cap from a broken bridge. Anything else is null: unknown.
const REACHY_SLOTS_RETRY_MS = 10000;
function readRelaySlots(header) {
  const match = /^(\d{1,3})\/(\d{1,3})$/.exec(header || "");
  if (!match || Number(match[2]) < 1) return null;
  const open = Number(match[1]), max = Number(match[2]);
  return {open, max, full: open >= max};
}

// Why a live sample got no answer when the reason lies with the server, not the source: the one
// inference slot is taken (by another tab, say: 429), the model backend is down or restarting
// (502/503), or nothing answered at all. The sample is skipped and the source keeps running.
// Anything else is a real failure, and null here.
function skippedSampleReason(err) {
  if (err?.status === 429) return "another inference is running";
  if (err?.status === 502 || err?.status === 503) return "the model backend is unavailable";
  if (err?.network === true) return "the Live Vision server is not reachable";
  return null;
}

// One status line from the bridge's /healthz. Only live (a fresh frame at the bridge) allows
// inference; anything else says why, in the bridge's own words where it gives them.
function describeReachyHealth(health) {
  if (!health || typeof health !== "object" || typeof health.state !== "string" || !health.state.trim()) {
    return {live: false, text: "bridge status unreadable"};
  }
  const stateName = health.state.trim().slice(0, 40);
  if (health.live === true) return {live: true, text: "live"};
  const reason = typeof health.reason === "string" ? health.reason.trim().slice(0, 300) : "";
  const why = reason || (stateName === "live" ? "no fresh frame in the last 5 s" : "");
  return {live: false, text: why ? `${stateName} · ${why}` : stateName};
}

const ADVANCED_DEFAULTS = {imageTokens: 512, imageTokenLimit: 512, topP: 1};

// A backend change retires both inference and health requests. Late responses
// from either must not update the new model's caption, settings or readiness.
class EngineRequestScope {
  constructor() { this.version = 0; this.requests = new Set(); }
  begin() {
    const ticket = {version: this.version, controller: new AbortController()};
    this.requests.add(ticket); return ticket;
  }
  current(ticket) { return ticket.version === this.version && !ticket.controller.signal.aborted; }
  release(ticket) { this.requests.delete(ticket); }
  invalidate() {
    this.version += 1;
    for (const ticket of this.requests) ticket.controller.abort();
    this.requests.clear();
  }
}

async function runEngineSwitch(scope, {before, request, after, refresh}) {
  scope.invalidate(); before();
  let failure;
  try { await request(); } catch (err) { failure = err; }
  finally {
    after();
    try { await refresh(); } catch (err) { failure ||= err; }
  }
  if (failure) throw failure;
}

function enginePolicy(active) {
  const policy = active?.request_policy;
  if (!policy && active?.model_id !== "brockone" && active?.id !== "brockone") return null;
  if (!policy || typeof policy.prompt !== "string" || !policy.prompt.trim() ||
      policy.max_tokens !== 64 || policy.temperature !== 0 || policy.image_tokens !== 512 || policy.stream !== false) {
    throw new Error("The selected model's fixed request policy is unavailable.");
  }
  return {...policy};
}

class EnginePolicySettings {
  constructor() { this.saved = null; }
  apply(policy, current) {
    if (policy) {
      this.saved ||= {...current};
      return {...current, prompt: policy.prompt, maxTokens: String(policy.max_tokens),
        imageTokenPreset: String(policy.image_tokens), customImageTokens: String(policy.image_tokens), topP: "1"};
    }
    const previous = this.saved; this.saved = null; return previous;
  }
}

function engineChoices(data) {
  if (!data || typeof data.configured !== "boolean" || !Array.isArray(data.engines)) throw new Error("Engine list unavailable");
  if (data.configured && (!data.active || typeof data.active.id !== "string" || typeof data.active.name !== "string")) throw new Error("Active engine identity unavailable");
  const choices = [], ids = new Set();
  for (const engine of data.engines) {
    if (!engine || typeof engine.id !== "string" || !engine.id || ids.has(engine.id) || typeof engine.name !== "string") {
      throw new Error("Invalid engine list");
    }
    ids.add(engine.id);
    choices.push({...engine, available: engine.available === true,
      reason: typeof engine.reason === "string" ? engine.reason : "Not available on this device"});
  }
  for (const id of ["brockone", "brocktwo"]) {
    if (!ids.has(id)) choices.push({id, name: id, available: false, reason: "Not configured on this device"});
  }
  return choices;
}

function renderEngineChoices(container, data, switching, onSwitch) {
  const nodes = engineChoices(data).map(engine => {
    const card = container.ownerDocument.createElement("div"); card.className = "model-choice";
    const button = container.ownerDocument.createElement("button"); button.type = "button";
    const active = engine.id === data.active?.id;
    button.textContent = engine.name; button.dataset.modelId = engine.id;
    button.className = active ? "active" : "";
    button.setAttribute("aria-pressed", String(active));
    button.disabled = switching || active || !engine.available;
    button.title = !engine.available ? engine.reason : (engine.profile || engine.name);
    button.addEventListener("click", () => { if (!button.disabled) onSwitch(engine.id); });
    const note = container.ownerDocument.createElement("small"); note.className = "model-availability";
    note.textContent = !engine.available ? engine.reason : active ? "Active" : "Available";
    card.append(button, note); return card;
  });
  container.replaceChildren(...nodes);
}

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

if (typeof module !== "undefined") module.exports = {SSEParser, readCompletionEvent, CAPTURE_PRESETS, FrameCadence, PROMPT_PRESETS, LatencySummary, appendTelemetrySample, telemetrySegments, ADVANCED_DEFAULTS, validateAdvancedSettings, readServerInferenceMs, readServerFirstTextMs, describeReachyHealth, readRelayStream, readRelaySlots, skippedSampleReason, REACHY_SAMPLE_MS, EngineRequestScope, EnginePolicySettings, enginePolicy, engineChoices, renderEngineChoices, runEngineSwitch};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const state = {ready: false, model: "", running: false, busy: false, media: null,
    abort: null, captureAt: null, completed: 0, imageURL: null, checking: false, cameraGeneration: 0,
    preset: "lightweight", frameCallback: null, sampled: 0, skipped: 0,
    liveStreaming: true, activeTrigger: null, engineId: null, timingGroup: 0,
    advanced: {...ADVANCED_DEFAULTS}, advancedValid: true, source: "camera",
    policy: null, activeEngineId: null, remoteSwitching: false};
  const engineRequests = new EngineRequestScope(), policySettings = new EnginePolicySettings();
  let engineSwitching = false, engineData = null, switchProgressTimer = null;
  // ?model=<id> (a bookmarkable/kiosk link) requests a switch once the backend is confirmed
  // ready, the same one-shot pattern as ?source=reachy below - honoured once, and only if
  // nothing else was chosen while the page was loading.
  let modelRequested = new URLSearchParams(window.location.search).get("model") || null;
  const switchingEngine = () => engineSwitching || state.remoteSwitching;
  const canvas = document.createElement("canvas");
  // The Reachy source while state.source === "reachy": bridge health, the relayed MJPEG
  // in #reachyImage, and the microphone. It shares running/cameraGeneration with the
  // camera, so the stop paths (Stop, preset change, a manual failure, pagehide) end it.
  // A failed live sample does not: see analyze().
  const reachy = {poll: null, polling: null, cadence: null, live: false, wasLive: false, healthAt: 0,
    audioLive: false, statusText: "checking the bridge…", streamToken: null, streamURL: null, openedAt: 0,
    streamOK: false, listening: false, audioURL: null, audioOpens: 0,
    streamRefused: false, slotsFull: 0, retryAt: 0};
  // /api/access: the HTTPS port for the camera link, and the token serve_ui.py wants on every
  // /reachy/ URL. Only this page can read it, which is what keeps other sites and rebound host
  // names off the robot's camera and microphone. Asked for at load; a failure is asked again by
  // the next Reachy poll, and so is a 401, which is how a restarted server refuses an old token.
  let access = null, accessRequest = null;
  function loadAccess() {
    if (access) return Promise.resolve(access);
    accessRequest ??= fetch("/api/access", {cache: "no-store", signal: AbortSignal.timeout(3000)})
      .then(async response => {
        const body = await response.json().catch(() => null);
        if (!response.ok) throw new Error(body?.error?.message || `Live Vision server returned HTTP ${response.status}.`);
        if (typeof body?.reachy_token !== "string" || !/^[A-Za-z0-9_-]{16,128}$/.test(body.reachy_token)) {
          throw new Error("Live Vision server sent no relay token.");
        }
        return access = body;
      })
      .finally(() => { accessRequest = null; });
    return accessRequest;
  }
  function reachyURL(path, params = {}) {
    return `${path}?${new URLSearchParams({...params, token: access.reachy_token})}`;
  }
  // ?source=reachy (the command centre's link) selects the robot once inference can run.
  let reachyRequested = new URLSearchParams(window.location.search).get("source") === "reachy";
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
    $("startButton").disabled = state.running || state.busy || switchingEngine();
    $("reachyButton").disabled = state.running || state.busy || switchingEngine();
    $("stopButton").disabled = !state.running && !state.busy;
    const sourceReady = state.running
      ? (state.source === "reachy" ? reachyFrameReady() : Boolean(state.media && $("video").readyState >= 2))
      : Boolean(state.imageURL);
    $("analyzeButton").disabled = !sourceReady || !state.ready || state.busy || !state.advancedValid || switchingEngine();
    for (const id of ["prompt", "promptPreset", "maxTokens", "imageTokenPreset", "customImageTokens", "topP", "lightweightPreset", "liveVlmPreset"]) {
      $(id).disabled = Boolean(state.policy) || switchingEngine();
    }
    $("liveToggleButton").setAttribute("aria-pressed", String(state.liveStreaming));
    $("liveToggleButton").textContent = `Live streaming: ${state.liveStreaming ? "On" : "Off"}`;
    $("listenButton").hidden = !reachyActive();
    $("listenButton").setAttribute("aria-pressed", String(reachy.listening));
    $("listenButton").textContent = `Listen: ${reachy.listening ? "On" : "Off"}`;
    // Flip camera only makes sense for the device's own webcam - Reachy's picture comes from the
    // robot regardless of which way a phone in your hand is facing.
    $("flipCameraButton").hidden = !(state.running && state.source === "camera");
  }
  function retireEngineAnswer(message) {
    state.abort?.abort(); state.abort = null; state.busy = false; state.activeTrigger = null;
    state.ready = false; state.completed = 0; state.captureAt = null;
    resetTimingGroup(message);
    $("answer").textContent = message; $("answer").classList.remove("streaming");
    $("runStatus").textContent = message; $("requestCount").textContent = "0 completed";
    for (const id of ["ttft", "totalTime", "frameAge"]) $(id).textContent = "—";
    $("captureStatus").textContent = "No frame sent";
  }
  function modelSettings() {
    return Object.fromEntries(["prompt", "maxTokens", "imageTokenPreset", "customImageTokens", "topP"].map(id => [id, $(id).value]));
  }
  function applyModelPolicy(policy, restored) {
    state.policy = policy;
    if (restored) for (const [id, value] of Object.entries(restored)) $(id).value = value;
    if (policy) state.advanced.imageTokenLimit = 512;
    applyAdvancedControls(false); matchPromptPreset();
    $("modelPolicyStatus").hidden = !policy;
    $("modelPolicyStatus").textContent = policy
      ? "brockone identifies Pokémon with a fixed prompt, 64 output tokens, 512 image tokens and temperature 0. Answers arrive complete; token timing is unavailable." : "";
    $("firstTextLabel").textContent = policy ? "Token timing unavailable" : "First visible token";
    $("serverTtftHelp").textContent = policy
      ? "brockone returns one complete answer. Native TTFT and token speed are not reported; Round trip is browser time to the complete response."
      : "TTFT: native inference start → first nonempty server text, including server scheduling.";
    $("presetDescription").hidden = Boolean(policy);
    controls();
  }
  async function checkBackend() {
    if (state.checking || engineSwitching) return;
    const ticket = engineRequests.begin(); state.checking = ticket;
    const load = async url => {
      try {
        const response = await fetch(url, {cache: "no-store", signal: AbortSignal.any([ticket.controller.signal, AbortSignal.timeout(5000)])});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
      } catch (_) { return null; }
    };
    try {
      const [healthData, modelsData, runtime, engines] = await Promise.all([
        load("/health/ready"), load("/v1/models"), load("/api/runtime"), load("/api/engines")]);
      if (!engineRequests.current(ticket) || state.checking !== ticket) return;
      if (!engines) throw new Error("Engine status unavailable");
      engineChoices(engines); engineData = engines;
      state.remoteSwitching = engines.switching === true;
      renderEngines();
      if (state.remoteSwitching) {
        retireEngineAnswer("Model switch in progress · camera preview stays connected");
        throw new Error("Model switch in progress");
      }
      const active = engines.configured ? engines.active : null;
      if (state.activeEngineId !== null && active?.id !== state.activeEngineId) {
        retireEngineAnswer("Model changed · waiting for a new answer");
      }
      state.activeEngineId = active?.id ?? null;
      const policy = enginePolicy(active);
      if (active?.available === false) throw new Error(active.reason || "Selected model is not qualified for inference");
      const model = modelsData?.data?.[0]?.id;
      if (healthData?.status !== "ready" || !model || (active?.model_id && model !== active.model_id)) throw new Error("Selected model is not ready");
      if (policy && ["prompt", "max_tokens", "temperature", "image_tokens", "stream"].some(key => runtime?.request_policy?.[key] !== policy[key])) throw new Error("Model policy and runtime do not match");
      const restored = policySettings.apply(policy, modelSettings());
      state.ready = true; state.model = model;
      $("backendStatus").textContent = "Local backend ready";
      $("backendStatus").className = "badge ready";
      $("modelName").textContent = model;
      try { applyRuntime(runtime); } catch (_) {
        $("staticClocksValue").textContent = "Unavailable";
        $("encoderCacheValue").textContent = "Unavailable";
        $("runtimeStatus").textContent = policy
          ? "brockone uses its fixed request policy. Device clocks and cache sizes are not reported by this endpoint."
          : "Engine settings unavailable. Input controls keep their current values; clocks and cache cannot be verified. Load defaults are shown above.";
      }
      applyModelPolicy(policy, restored);
    } catch (err) {
      if (!engineRequests.current(ticket) || state.checking !== ticket) return;
      state.ready = false;
      $("backendStatus").textContent = state.remoteSwitching ? "Switching model…" : "Backend not ready";
      $("backendStatus").className = "badge unavailable";
      $("modelName").textContent = "Waiting for local TensorRT-Edge-LLM";
      $("staticClocksValue").textContent = "Unavailable"; $("encoderCacheValue").textContent = "Unavailable";
      $("runtimeStatus").textContent = "Waiting for the backend to report active engine settings. Load defaults are shown above.";
      if (!state.busy) $("runStatus").textContent = err.message || "Waiting for local backend";
    } finally {
      engineRequests.release(ticket);
      if (state.checking === ticket) { state.checking = null; controls(); }
    }
    // Honoured once, and only if nothing else was chosen while the backend was loading.
    if (state.ready && reachyRequested) {
      reachyRequested = false;
      if (!state.running && !state.busy && !state.imageURL) startReachy();
    }
    if (state.ready && modelRequested && engineData?.configured) {
      const target = modelRequested; modelRequested = null;
      const choice = engineChoices(engineData).find(engine => engine.id === target);
      if (!choice) error(`?model=${target} is not a known model on this device.`);
      else if (target !== engineData.active?.id) {
        if (choice.available) switchToEngine(target);
        else error(`?model=${target} requested, but it is not available: ${choice.reason || "not available on this device"}`);
      }
    }
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
    if (state.busy || !state.ready || !state.advancedValid || switchingEngine()) return;
    // Never send the robot's last picture as if it were current; the status says why not.
    if (source === $("reachyImage") && !reachyFrameReady()) { renderReachySampling(); controls(); return; }
    // A live Reachy sample never ends the source: the robot's video and microphone stay open
    // through a bad sample, and the next one tries again. Only Stop, a preset change, pagehide
    // or a failed manual request do. The camera keeps stopping on a real failure, as before;
    // both skip what the server turned away for now (skippedSampleReason).
    const keepSource = trigger === "live" && state.source === "reachy";
    const policy = state.policy;
    const prompt = policy?.prompt ?? $("prompt").value.trim();
    const maxTokens = policy?.max_tokens ?? Number($("maxTokens").value);
    if (!prompt || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) {
      error("Enter a prompt and an output token limit from 1 to 512.");
      if (!keepSource) stop();
      return;
    }
    // Freeze request settings and group ownership before capture. Later UI or
    // engine changes affect the next request and cannot mix timing populations.
    let advanced;
    try { advanced = advancedValues(); } catch (err) { error(err.message); return; }
    if (policy) advanced = {topP: 1, imageTokens: policy.image_tokens};
    const timingGroup = state.timingGroup;
    const temperature = policy?.temperature ?? CAPTURE_PRESETS[state.preset].temperature;
    const ticket = engineRequests.begin(), controller = ticket.controller;
    state.busy = true; state.abort = controller; state.activeTrigger = trigger; controls(); error();
    $("ttft").textContent = "—"; $("totalTime").textContent = "—";
    $("runStatus").textContent = "Reading frame…";
    let reader, counted = false;
    // Nothing answered at all: the network failed, not this request (see skippedSampleReason).
    const network = err => { if (err?.name === "TypeError") err.network = true; throw err; };
    try {
      const image = capture(source);
      state.captureAt = image.capturedAt;
      counted = state.running && source === liveSource();
      if (counted) state.sampled += 1;
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
      }).catch(network);
      controller.signal.throwIfAborted();
      if (state.abort !== controller || !engineRequests.current(ticket)) return;
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw Object.assign(new Error(body.error?.message || `Backend returned HTTP ${response.status}.`),
          {status: response.status});
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
          if (firstToken === null) { firstToken = performance.now(); if (!policy) $("ttft").textContent = duration(firstToken - started); }
          output += text;
          if (output.length > 65536) throw new Error("Model output exceeded the display limit.");
          $("answer").textContent = output;
          $("answer").scrollTop = $("answer").scrollHeight;
          if (!policy) $("answer").classList.add("streaming");
          $("runStatus").textContent = policy ? "Answer received" : "Writing answer…";
        }
      });
      reader = response.body.getReader();
      while (!done) {
        const chunk = await reader.read().catch(network);
        controller.signal.throwIfAborted();
        if (state.abort !== controller || !engineRequests.current(ticket)) return;
        if (chunk.done) { parser.feed(decoder.decode(), true); break; }
        parser.feed(decoder.decode(chunk.value, {stream: true}));
      }
      if (!done) throw new Error("Connection ended before the model completed its answer.");
      if (!["stop", "length"].includes(finishReason)) throw new Error("Backend stream did not confirm a successful completion.");
      if (!output.trim()) throw new Error("The backend completed without visible answer text. Try a larger output token limit.");
      $("totalTime").textContent = duration(performance.now() - started);
      if (timingGroup === state.timingGroup) {
        serverFirstTextMs = policy ? null : readServerFirstTextMs(serverMetrics);
        const milliseconds = policy ? null : readServerInferenceMs(serverMetrics);
        if (milliseconds === null) {
          latency.lastMs = null;
          $("timingStatus").textContent = policy ? "Complete answer received · browser round trip only; native token timings unavailable."
            : "Latest answer has no valid server inference timing; excluded from Average and Timed.";
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
      const skipped = trigger === "live" ? skippedSampleReason(err) : null;
      if (err.name === "AbortError" || controller.signal.aborted) $("runStatus").textContent = "Stopped · partial answer";
      else if (skipped) {
        // Counted as skipped, not sent: it got no answer. Video, audio and sampling go on.
        if (counted) state.sampled -= 1;
        state.skipped += 1;
        $("runStatus").textContent = `Skipped · ${skipped}`;
        renderLiveCounts();
      } else {
        error(err.message); $("runStatus").textContent = "Request failed";
        if (!keepSource) state.running = false;
      }
    } finally {
      engineRequests.release(ticket);
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
  // Lightweight cadence for either live source: capture after each answer, at most once
  // per interval. A Reachy frame only counts as ready while the bridge reports live.
  async function cameraLoop(generation) {
    while (state.running && generation === state.cameraGeneration) {
      const start = performance.now();
      if (state.liveStreaming && state.ready && liveFrameReady()) await analyze(liveSource(), "live");
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
  // Live VLM WebUI cadence for Reachy: the frame-counting loop above needs a <video>, so this
  // samples every REACHY_SAMPLE_MS instead, still skipping (and counting) samples while busy.
  // Ticks while the bridge is not live send nothing and are not counted as skipped.
  function reachyCadenceLoop(generation) {
    function tick() {
      reachy.cadence = null;
      if (!state.running || generation !== state.cameraGeneration) return;
      if (state.liveStreaming && reachyFrameReady()) {
        if (state.busy) state.skipped += 1;
        else if (state.ready) void analyze($("reachyImage"), "live");
      }
      renderReachySampling();
      reachy.cadence = setTimeout(tick, REACHY_SAMPLE_MS);
    }
    reachy.cadence = setTimeout(tick, REACHY_SAMPLE_MS);
  }
  function reachyActive() { return state.running && state.source === "reachy"; }
  function liveSource() { return state.source === "reachy" ? $("reachyImage") : $("video"); }
  function liveFrameReady() { return state.source === "reachy" ? reachyFrameReady() : $("video").readyState >= 2; }
  // Only a frame this connection delivered (load) while the bridge is live and the relay still
  // has the stream: never the last picture of a stream that has quietly ended.
  function reachyFrameReady() {
    return reachy.live && performance.now() - reachy.healthAt < REACHY_HEALTH_STALE_MS &&
      reachy.streamURL !== null && reachy.streamOK && $("reachyImage").naturalWidth > 0;
  }
  function openReachyStream() {
    // Without the relay token (a poll has just dropped a stale one) the next poll opens it.
    if (!access) return;
    // A random token per connection names it when the health poll asks the relay about it,
    // and makes the URL new, so the browser really fetches rather than reusing an old stream.
    // Replacing src also ends the previous connection.
    reachy.streamToken = Array.from(crypto.getRandomValues(new Uint32Array(3)),
      word => word.toString(36).padStart(7, "0")).join("");
    reachy.streamURL = reachyURL("/reachy/mjpeg", {stream: reachy.streamToken});
    reachy.openedAt = performance.now(); reachy.streamOK = false;
    $("reachyImage").src = reachy.streamURL;
  }
  function closeReachyStream() {
    reachy.streamToken = null; reachy.streamURL = null; reachy.streamOK = false;
    // Clearing src is what ends an <img>'s MJPEG request, so the relay and the bridge let go.
    $("reachyImage").src = "";
  }
  function openReachyAudio() {
    if (!access) return;   // As for the video: the next poll that reports audio opens it.
    const url = reachy.audioURL = reachyURL("/reachy/audio.mp3", {open: ++reachy.audioOpens});
    $("reachyAudio").src = url;
    $("reachyAudio").play().catch(err => {
      // Stop and Listen off interrupt play() on purpose. A failed stream is retried by the
      // next health poll that reports microphone audio; a blocked one needs the user.
      if (reachy.audioURL !== url || err.name === "AbortError") return;
      reachy.audioURL = null;
      if (err.name === "NotAllowedError") {
        reachy.listening = false;
        error("This browser blocked audio playback. Press Listen again to allow it.");
      }
      renderReachyStatus(); controls();
    });
  }
  function closeReachyAudio() {
    reachy.audioURL = null;
    const audio = $("reachyAudio");
    // Removing src and reloading is what aborts a media fetch. The bridge only runs its
    // MP3 encoder while somebody listens, so turning Listen off must really disconnect.
    audio.pause(); audio.removeAttribute("src"); audio.load();
  }
  function renderReachyStatus() {
    if (!reachyActive()) return;
    const image = $("reachyImage");
    // The <img> keeps its last frame when the robot goes quiet; do not let it look live.
    image.classList.toggle("stale", !reachyFrameReady());
    const parts = ["Reachy Mini", reachy.statusText];
    if (reachy.live) {
      if (document.visibilityState !== "visible") parts.push("video paused while this tab is hidden");
      else if (reachy.slotsFull && !reachy.streamOK) {
        parts.push(`All ${reachy.slotsFull} Reachy streams on this server are in use – close another Live Vision tab or turn Listen off`);
      } else if (reachy.streamURL === null) parts.push("video reconnecting");
      else if (!reachy.streamOK || !image.naturalWidth) parts.push("waiting for video");
      else parts.push(`${image.naturalWidth}×${image.naturalHeight}`);
    }
    if (reachy.listening) {
      parts.push(reachy.audioURL === null ? "audio reconnecting" : reachy.audioLive ? "listening" : "listening · microphone silent");
    }
    $("sourceStatus").textContent = parts.join(" · ");
  }
  function renderReachySampling() {
    if (!reachyActive()) return;
    let text;
    if (!reachyFrameReady()) {
      text = !reachy.healthAt ? "Checking the Reachy bridge"
        : reachy.live ? "Not sending · waiting for Reachy video" : "Not sending · Reachy Mini is not live";
    } else if (!state.liveStreaming) text = "Live streaming off · use Run inference";
    else if (state.preset === "live-vlm") {
      text = `${state.sampled} frames sent · ${state.skipped} skipped while busy · every ${REACHY_SAMPLE_MS / 1000} s, time-based (30 frames at the bridge's ~${REACHY_BRIDGE_FPS} FPS)`;
    } else text = "Capture after each answer";
    $("samplingStatus").textContent = text;
  }
  // The sampling line after a skipped live sample; each live loop otherwise writes its own.
  function renderLiveCounts() {
    if (state.source === "reachy") renderReachySampling();
    else if (state.preset === "live-vlm" && state.liveStreaming) {
      $("samplingStatus").textContent = `${state.sampled} frames sent · ${state.skipped} skipped while busy`;
    }
  }
  function applyReachyHealth(health, failure, relay, slots, stream, askedAt) {
    const status = failure ? {live: false, text: failure} : describeReachyHealth(health);
    const now = performance.now();
    reachy.live = status.live; reachy.statusText = status.text; reachy.healthAt = now;
    reachy.audioLive = !failure && health?.audio?.live === true;
    // A hidden tab keeps its video closed (visibilitychange below); it reopens when shown.
    if (status.live && document.visibilityState === "visible") {
      // The relay's answer is about the stream this poll named; a reopen since makes it moot.
      const judged = relay && stream === reachy.streamToken && askedAt - reachy.openedAt > REACHY_STREAM_SETTLE_MS;
      const dead = judged && (!relay.open || relay.idleMs > REACHY_STREAM_IDLE_MS);
      if (dead) reachy.streamOK = false;
      const stillborn = !reachy.streamOK && now - reachy.openedAt > REACHY_STREAM_GRACE_MS;
      // The <img> was refused while every stream slot was taken: say so, and try again in 10 s
      // rather than at every poll, each of which would only be turned away again.
      if (reachy.streamRefused) {
        reachy.streamRefused = false;
        reachy.slotsFull = slots?.full ? slots.max : 0;
        if (slots?.full) reachy.retryAt = now + REACHY_SLOTS_RETRY_MS;
      }
      // Reopen when the bridge comes back (the relay drops a stream silent for 20 s), when the
      // stream failed or went quiet, or when a new one delivered no frame within the grace.
      if (now >= reachy.retryAt && (!reachy.wasLive || reachy.streamURL === null || dead || stillborn)) {
        openReachyStream();
      }
    }
    reachy.wasLive = status.live;
    if (reachy.listening && reachy.audioURL === null && reachy.audioLive) openReachyAudio();
    $("liveIndicator").hidden = !status.live;
    renderReachyStatus(); renderReachySampling(); controls();
  }
  async function pollReachy(generation) {
    // Hidden, the page has closed its video; only Listen still needs the bridge's state.
    if (reachy.polling || (document.visibilityState !== "visible" && !reachy.listening)) return;
    const poll = reachy.polling = {};
    const stream = reachy.streamToken, askedAt = performance.now();
    let health = null, failure = null, relay = null, slots = null;
    try {
      await loadAccess();
      const response = await fetch(reachyURL("/reachy/healthz", stream ? {stream} : {}),
        {cache: "no-store", signal: AbortSignal.timeout(2500)});
      const body = await response.json().catch(() => null);
      if (response.status === 401) {
        // This server restarted since the page read its token: drop it, the next poll asks again.
        access = null;
        throw new Error("Live Vision server restarted · reconnecting");
      }
      if (!response.ok) throw new Error(body?.error?.message || `Live Vision server returned HTTP ${response.status}.`);
      health = body; relay = readRelayStream(response.headers.get("X-Reachy-Stream"));
      slots = readRelaySlots(response.headers.get("X-Reachy-Slots"));
    } catch (err) {
      failure = err.name === "TimeoutError" ? "bridge did not answer in time"
        : err instanceof TypeError ? "Live Vision server not reachable" : err.message;
    } finally {
      if (reachy.polling === poll) reachy.polling = null;
    }
    if (!reachyActive() || generation !== state.cameraGeneration) return;
    applyReachyHealth(health, failure, relay, slots, stream, askedAt);
  }
  function startReachy() {
    // No getUserMedia: the robot's video comes from the Jetson, so this works over plain HTTP.
    error(); reachyRequested = false;
    // Same reasoning as startCamera()'s own releaseCamera() call: switching source while one is
    // already open otherwise leaks the previous one instead of properly stopping it - a webcam's
    // MediaStream tracks were never being stopped when switching straight to Reachy, which could
    // leave the camera hardware held (and on some browsers, unavailable to reacquire later).
    releaseCamera();
    state.running = true; state.source = "reachy";
    const generation = ++state.cameraGeneration;
    Object.assign(reachy, {polling: null, live: false, wasLive: false, healthAt: 0, audioLive: false,
      statusText: "checking the bridge…", openedAt: 0, streamOK: false, listening: false,
      streamRefused: false, slotsFull: 0, retryAt: 0});
    state.sampled = 0; state.skipped = 0;
    $("video").hidden = true; $("uploadedImage").hidden = true; $("placeholder").hidden = true;
    $("reachyImage").hidden = false;
    renderReachyStatus(); renderReachySampling(); controls();
    reachy.poll = setInterval(() => pollReachy(generation), REACHY_HEALTH_MS);
    void pollReachy(generation);
    if (state.preset === "live-vlm") reachyCadenceLoop(generation); else cameraLoop(generation);
  }
  // Safe to call repeatedly: an aborted request's cleanup releases a second time.
  function releaseReachy() {
    const active = reachy.poll !== null;
    clearInterval(reachy.poll); reachy.poll = null;
    clearTimeout(reachy.cadence); reachy.cadence = null;
    Object.assign(reachy, {polling: null, live: false, wasLive: false, listening: false,
      streamRefused: false, slotsFull: 0, retryAt: 0});
    if (reachy.streamURL !== null || $("reachyImage").getAttribute("src")) closeReachyStream();
    if (reachy.audioURL !== null || $("reachyAudio").hasAttribute("src")) closeReachyAudio();
    if (active) {
      $("reachyImage").hidden = true;
      if (state.imageURL) $("uploadedImage").hidden = false; else $("placeholder").hidden = false;
    }
  }
  function releaseCamera() {
    if (state.frameCallback !== null) $("video").cancelVideoFrameCallback?.(state.frameCallback);
    state.frameCallback = null;
    if (state.media) state.media.getTracks().forEach(track => track.stop());
    state.media = null; $("video").srcObject = null; $("liveIndicator").hidden = true;
    releaseReachy();
    $("sourceStatus").textContent = state.imageURL ? "Selected image"
      : state.source === "reachy" ? "Reachy Mini stopped" : "Camera stopped";
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
  // Outward-facing by default (environment): matches what this UI is normally pointed at (a
  // scene, not the person holding the phone). "ideal" not "exact": a laptop with one camera (no
  // facingMode at all) still gets a stream instead of a hard getUserMedia failure.
  let facingMode = "environment";
  async function startCamera() {
    error(); reachyRequested = false;
    if (!navigator.mediaDevices?.getUserMedia) { error("Camera access needs HTTPS or http://localhost. You can choose an image instead."); return; }
    if (state.preset === "live-vlm" && !$("video").requestVideoFrameCallback) {
      error("This browser cannot count video frames. Use a current browser or choose Lightweight."); return;
    }
    releaseCamera();  // drop any existing stream first - flipping cameras while one is open can
                       // otherwise ask a phone to hold two camera handles at once and fail.
    state.running = true; state.source = "camera"; const generation = ++state.cameraGeneration; controls();
    try {
      const media = await navigator.mediaDevices.getUserMedia({audio: false,
        video: {...CAPTURE_PRESETS[state.preset].cameraConstraints, facingMode: {ideal: facingMode}}});
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
  }
  $("startButton").addEventListener("click", startCamera);
  $("flipCameraButton").addEventListener("click", () => {
    facingMode = facingMode === "environment" ? "user" : "environment";
    if (state.running && state.source === "camera") startCamera();
  });
  // No !state.running guard (unlike the old version of this handler): startReachy() now tears
  // down any active webcam itself via releaseCamera(), the same way startButton's startCamera()
  // has always torn down an active Reachy session - so this needs to work while the camera is
  // already running, not only from a stopped state. That asymmetry (camera->Reachy silently did
  // nothing unless Stop was clicked first; Reachy->camera always worked) was the reported bug.
  $("reachyButton").addEventListener("click", () => { if (!state.busy) startReachy(); });
  $("listenButton").addEventListener("click", () => {
    if (!reachyActive()) return;
    error();
    reachy.listening = !reachy.listening;
    if (reachy.listening) openReachyAudio(); else closeReachyAudio();
    renderReachyStatus(); controls();
  });
  $("reachyImage").addEventListener("load", () => {
    // The first frame of this connection: proof that it delivers. Until the relay says
    // otherwise, the picture it shows is current.
    if (!reachyActive() || $("reachyImage").getAttribute("src") !== reachy.streamURL) return;
    reachy.streamOK = true; reachy.slotsFull = 0;
    renderReachyStatus(); renderReachySampling(); controls();
  });
  $("reachyImage").addEventListener("error", () => {
    // src = "" on Stop fires this too; only the current stream failing counts. Its cause
    // (bridge down, stream cap, a restarted server) is not visible here, so the next poll
    // retries, and its X-Reachy-Slots says whether the cap was why.
    if (!reachyActive() || reachy.streamURL === null || $("reachyImage").getAttribute("src") !== reachy.streamURL) return;
    reachy.streamToken = null; reachy.streamURL = null; reachy.streamOK = false; reachy.streamRefused = true;
    renderReachyStatus(); renderReachySampling(); controls();
  });
  // A hidden tab shows nobody the robot's video, but its MJPEG stream would go on holding one of
  // the server's few stream slots. Close it, which also pauses live inference: with no stream
  // there is no frame to send. Listen keeps its audio playing, which is what it is for with the
  // tab in the background. Shown again, a poll now reopens the video.
  document.addEventListener("visibilitychange", () => {
    if (!reachyActive()) return;
    if (document.visibilityState === "visible") { void pollReachy(state.cameraGeneration); return; }
    if (reachy.streamURL !== null) closeReachyStream();
    renderReachyStatus(); renderReachySampling(); controls();
  });
  for (const type of ["error", "ended"]) {
    $("reachyAudio").addEventListener(type, () => {
      // An idle relay ends the stream after 20 s of silence; reopened when audio returns.
      if (!reachyActive() || reachy.audioURL === null || $("reachyAudio").getAttribute("src") !== reachy.audioURL) return;
      reachy.audioURL = null;
      renderReachyStatus();
    });
  }
  $("stopButton").addEventListener("click", stop);
  $("analyzeButton").addEventListener("click", () => analyze(state.running ? liveSource() : $("uploadedImage")));
  $("liveToggleButton").addEventListener("click", () => {
    state.liveStreaming = !state.liveStreaming;
    // Pause automatic requests without stopping the preview or interrupting
    // an explicitly requested manual inference.
    if (!state.liveStreaming && state.activeTrigger === "live") state.abort?.abort();
    $("samplingStatus").textContent = state.liveStreaming
      ? (state.preset === "live-vlm" ? "Every 30 frames · live streaming on" : "Capture after each answer")
      : "Live streaming off · use Run inference";
    renderReachySampling();
    controls();
  });
  $("imageInput").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    reachyRequested = false;
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
  if (reachyRequested) $("sourceStatus").textContent = "Reachy Mini · starts when the local backend is ready";
  setInterval(checkBackend, 5000); checkBackend(); controls();
  // Read once at load: the relay token, and over plain HTTP the camera's HTTPS link. If this
  // fails, the Reachy poll asks again when it needs the token; the link is not retried.
  const accessLoaded = loadAccess();
  accessLoaded.catch(() => {});
  if (!window.isSecureContext) {
    $("cameraHelp").textContent = "Image upload and Reachy Mini work here. Camera access requires HTTPS.";
    accessLoaded.then(info => {
      if (!Number.isInteger(info.https_port) || info.https_port < 1 || info.https_port > 65535) return;
      const url = new URL(window.location.href);
      url.protocol = "https:"; url.port = String(info.https_port); url.pathname = "/"; url.search = ""; url.hash = "";
      const link = document.createElement("a");
      link.href = url.href; link.textContent = "Open HTTPS for camera access";
      $("cameraHelp").append(" ", link, ". This device uses a local certificate.");
    }).catch(() => {});
  }
  startDeviceTelemetry();

  // Reachy Mini motor/app/speech controls: talks to /api/reachy/* (this server's own routes onto
  // the robot's daemon - see serve_ui.py's reachy_control), separate from the /reachy/ camera+mic
  // relay used elsewhere in this file. Polled rather than tied to the camera source, so the panel
  // works whether the robot is the active input or just sitting there reachable.
  async function reachyControlRequest(path, options) {
    try {
      const response = await fetch(path, options);
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.error?.message || body.message || `HTTP ${response.status}`);
      if (body.ok === false) throw new Error(body.message || "request failed");
      return body;
    } catch (err) {
      error(`Reachy: ${err.message}`);
      throw err;
    }
  }
  function reachyPost(path, jsonBody) {
    const options = {method: "POST"};
    if (jsonBody !== undefined) { options.headers = {"Content-Type": "application/json"}; options.body = JSON.stringify(jsonBody); }
    return reachyControlRequest(path, options);
  }
  // Live pose control: X/Y, Z, roll/pitch/yaw, body_yaw and antennas, all POSTed together to
  // /api/reachy/target (serve_ui.py's reachy_control forwards them straight into Reachy.set_target).
  // Throttled to one POST per 60ms so a fast drag doesn't flood the daemon; syncCtl() pulls the
  // live pose back into the widgets, but only once 1500ms have passed since the user last touched
  // a control, so a poll reply never yanks a slider out from under a finger mid-drag.
  const T = {x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0, body_yaw: 0, antennas: [0.175, 0.175]};
  let tBusy = false, tPend = false, tLast = 0, tGrabbed = 0;
  const D2R = Math.PI / 180;
  function tTouch() { tGrabbed = Date.now(); }
  async function tSend() {
    if (tBusy) { tPend = true; return; }
    const now = Date.now();
    if (now - tLast < 60) { if (!tPend) { tPend = true; setTimeout(() => { tPend = false; tSend(); }, 60 - (now - tLast)); } return; }
    tBusy = true; tLast = now;
    const status = $("reachyPoseStatus");
    try {
      await reachyPost("/api/reachy/target", T);
      status.hidden = true;
    } catch (e) {
      // The shared error() banner (also set by reachyPost above) is transient and gets
      // overwritten by unrelated polling within seconds - a drag that silently stops moving the
      // robot (motors switched to Limp mid-demo, most often) needs a reason that stays put next
      // to the controls, not a flash of red text elsewhere on the page.
      status.textContent = e.message || "Move refused";
      status.hidden = false;
    }
    tBusy = false;
    if (tPend) { tPend = false; tSend(); }
  }
  function rcNum(v, n) { return (v < 0 ? "" : " ") + v.toFixed(n === undefined ? 3 : n); }
  function bindSlider(id, lblId, key, unit, idx) {
    const el = $(id), lbl = $(lblId);
    if (!el) return;
    const paint = () => { const v = parseFloat(el.value); lbl.textContent = rcNum(v) + (unit || ""); };
    el.addEventListener("input", () => {
      tTouch();
      const v = parseFloat(el.value);
      if (idx === undefined) T[key] = v; else T.antennas[idx] = v;
      paint(); tSend();
    });
    paint();
  }
  // A pad maps the two axes of a square onto two target fields. `inv` flips the vertical axis for
  // pitch, which is positive downward on this robot: without it, dragging up would look down.
  function bindPad(id, lblId, kx, ky, rx, ry, invX, invY) {
    const pad = $(id); if (!pad) return;
    const dot = pad.querySelector("i"), lbl = $(lblId);
    let down = false;
    const paint = () => {
      const fx = ((invX ? -T[kx] : T[kx]) / rx + 1) / 2, fy = ((invY ? -T[ky] : T[ky]) / ry + 1) / 2;
      dot.style.left = (Math.min(1, Math.max(0, fx)) * 100) + "%";
      dot.style.top = (100 - Math.min(1, Math.max(0, fy)) * 100) + "%";
      lbl.textContent = rcNum(T[kx]) + " " + rcNum(T[ky]);
    };
    const at = ev => {
      const b = pad.getBoundingClientRect();
      const fx = Math.min(1, Math.max(0, (ev.clientX - b.left) / b.width));
      const fy = Math.min(1, Math.max(0, (ev.clientY - b.top) / b.height));
      const vx = (fx * 2 - 1) * rx, vy = ((1 - fy) * 2 - 1) * ry;
      T[kx] = invX ? -vx : vx;
      T[ky] = invY ? -vy : vy;
      tTouch(); paint(); tSend();
    };
    pad.addEventListener("pointerdown", e => { down = true; pad.setPointerCapture(e.pointerId); at(e); });
    pad.addEventListener("pointermove", e => { if (down) at(e); });
    pad.addEventListener("pointerup", e => { down = false; at(e); });
    pad.addEventListener("pointercancel", () => { down = false; });
    pad._paint = paint; paint();
  }
  // A small fixed step per press/tick, held to repeat - independent of the pad's own drag
  // sensitivity, for a finer, steadier adjustment than a twitchy pad allows. sign follows the
  // same invX/invY convention bindPad above was just given: a jog button moves the same way as
  // dragging the pad in that direction would.
  function bindJog(id, key, stepDeg, sign, limitRad, padId) {
    const btn = $(id); if (!btn) return;
    let timer = null;
    const tick = () => {
      T[key] = Math.max(-limitRad, Math.min(limitRad, T[key] + sign * stepDeg * D2R));
      tTouch();
      const pad = $(padId); if (pad && pad._paint) pad._paint();
      tSend();
    };
    const start = ev => { ev.preventDefault(); tick(); timer = setInterval(tick, 120); };
    const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
    btn.addEventListener("pointerdown", start);
    btn.addEventListener("pointerup", stop);
    btn.addEventListener("pointerleave", stop);
    btn.addEventListener("pointercancel", stop);
  }
  // Body yaw drives the head yaw target to match (so the two swing together as one turn, rather
  // than the base rotating while the head stays fixed relative to it - awkward to watch and to
  // demo), and snaps near a handful of key angles: an in-zone drag holds at the snap point
  // instead of tracking the pointer 1:1, the way a detent does, so a fixed angle is easy to hit
  // exactly rather than eyeballed against the readout.
  const BODY_YAW_SNAPS_DEG = [-135, -90, -45, 0, 45, 90, 135];
  const BODY_YAW_SNAP_TOLERANCE_DEG = 2.5;
  function snapBodyYawRad(rad) {
    const deg = rad / D2R;
    let nearest = null, nearestDist = Infinity;
    for (const snap of BODY_YAW_SNAPS_DEG) {
      const dist = Math.abs(deg - snap);
      if (dist < nearestDist) { nearest = snap; nearestDist = dist; }
    }
    return nearestDist <= BODY_YAW_SNAP_TOLERANCE_DEG ? nearest * D2R : rad;
  }
  function bindBodyYaw(id, lblId) {
    const el = $(id), lbl = $(lblId);
    if (!el) return;
    const paint = () => { lbl.textContent = rcNum(T.body_yaw) + " rad"; };
    el.addEventListener("input", () => {
      tTouch();
      const snapped = snapBodyYawRad(parseFloat(el.value));
      el.value = snapped;
      T.body_yaw = snapped;
      T.yaw = Math.max(-Math.PI, Math.min(Math.PI, snapped));
      paint();
      const pad = $("padPY"); if (pad && pad._paint) pad._paint();
      tSend();
    });
    paint();
  }
  function syncCtl(rs) {
    if (Date.now() - tGrabbed < 1500) return;
    const p = rs.pose_deg || {}, m = rs.pos_m || {};
    const set = (id, lbl, v, unit) => {
      const e = $(id);
      if (e && document.activeElement !== e && v != null) {
        e.value = v;
        const b = $(lbl); if (b) b.textContent = rcNum(v) + (unit || "");
      }
    };
    if (p.roll != null) { T.roll = p.roll * D2R; set("roll", "rollv", T.roll, " rad"); }
    if (p.pitch != null) T.pitch = p.pitch * D2R;
    if (p.yaw != null) T.yaw = p.yaw * D2R;
    if (rs.body_yaw_deg != null) { T.body_yaw = rs.body_yaw_deg * D2R; set("byaw", "byawv", T.body_yaw, " rad"); }
    if (m.x != null) T.x = m.x;
    if (m.y != null) T.y = m.y;
    if (m.z != null) { T.z = m.z; set("posZ", "zv", T.z, ""); }
    const a = rs.antennas_deg;
    if (a && a.length === 2) {
      T.antennas = [a[0] * D2R, a[1] * D2R];
      set("antL", "antLv", T.antennas[0], " rad"); set("antR", "antRv", T.antennas[1], " rad");
    }
    const px = $("padXY"), pp = $("padPY");
    if (px && px._paint) px._paint();
    if (pp && pp._paint) pp._paint();
  }
  bindSlider("antL", "antLv", null, " rad", 0);
  bindSlider("antR", "antRv", null, " rad", 1);
  bindSlider("roll", "rollv", "roll", " rad");
  bindBodyYaw("byaw", "byawv");
  bindSlider("posZ", "zv", "z", "");
  bindPad("padXY", "xyv", "x", "y", 0.02, 0.02, false, false);
  // Both axes flipped from the original porch-feed mapping - reported backwards on the physical
  // robot for set_target's target_head_pose (a different daemon route than the goto()-based
  // look()/centre() this sign convention was originally verified against - see Reachy.look()'s
  // own docstring - so the two need not agree).
  bindPad("padPY", "pyv", "yaw", "pitch", Math.PI, 0.7, true, false);
  bindJog("jogPitchUp", "pitch", 2, 1, 0.7, "padPY");
  bindJog("jogPitchDown", "pitch", 2, -1, 0.7, "padPY");
  bindJog("jogYawLeft", "yaw", 2, 1, Math.PI, "padPY");
  bindJog("jogYawRight", "yaw", 2, -1, Math.PI, "padPY");

  let reachyAppsLoadedAt = 0;
  async function refreshReachyApps() {
    if (Date.now() - reachyAppsLoadedAt < 15000) return;
    reachyAppsLoadedAt = Date.now();
    try {
      const apps = await reachyControlRequest("/api/reachy/apps");
      const select = $("reachyAppSelect");
      const installed = apps.installed || [];
      select.innerHTML = installed.length
        ? installed.map(name => `<option value="${name}">${name}${name === apps.current ? " (running)" : ""}</option>`).join("")
        : `<option value="">No apps found</option>`;
    } catch (_) { /* surfaced already via error() */ }
  }
  async function pollReachyControlState() {
    try {
      const response = await fetch("/api/reachy/state", {cache: "no-store"});
      const st = await response.json();
      const panel = $("reachyControls");
      // Available whenever the robot is configured, regardless of which feed is active - these
      // control the robot itself (motors, apps, speech), not the video source, and hiding them
      // just because the webcam is on made TTS unreachable without switching feeds first. A
      // <details>, collapsed by default, is the safety net instead: nothing here fires without a
      // deliberate expand-then-click.
      if (!st.enabled) { panel.hidden = true; return; }
      panel.hidden = false;
      $("reachyControlStatus").textContent = st.reachable === false
        ? "Robot daemon unreachable" : `Motors: ${st.motor_mode || "unknown"}`;
      for (const [id, mode] of MOTOR_BUTTONS) $(id).classList.toggle("active", st.motor_mode === mode);
      if (st.reachable !== false) { refreshReachyApps(); syncCtl(st); }
    } catch (_) { /* keep last known state on a transient poll failure */ }
  }
  $("reachyWake").addEventListener("click", () => reachyPost("/api/reachy/action/wake"));
  $("reachySleep").addEventListener("click", () => reachyPost("/api/reachy/action/sleep"));
  $("reachyCenter").addEventListener("click", () => reachyPost("/api/reachy/action/center"));
  $("reachyFaceSound").addEventListener("click", () => reachyPost("/api/reachy/action/look-at-voice"));
  // Stiff/Soft/Limp, not a dropdown - the wording and routes porch-feed's own command centre
  // uses (main, nvr/feed/porch_feed.py). Centre and the pose pad/sliders below silently refuse
  // every move while motors are Limp (see Reachy.set_target's own "motors are disabled" check in
  // nvr/reachy/reachy.py), so this needs to read as a mode to actively pick, not a setting to
  // notice was wrong afterward.
  const MOTOR_BUTTONS = [["reachyMotorStiff", "enabled"], ["reachyMotorSoft", "gravity_compensation"], ["reachyMotorLimp", "disabled"]];
  for (const [id, mode] of MOTOR_BUTTONS) $(id).addEventListener("click", () => reachyPost(`/api/reachy/motors/${mode}`));
  $("reachyAppStart").addEventListener("click", () => {
    const name = $("reachyAppSelect").value;
    if (name) reachyPost(`/api/reachy/apps/start/${encodeURIComponent(name)}`);
  });
  $("reachyAppStop").addEventListener("click", () => reachyPost("/api/reachy/apps/stop"));
  $("reachySpeakerVol").addEventListener("change", e => reachyPost(`/api/reachy/volume/speaker/${e.target.value}`));
  $("reachyMicVol").addEventListener("change", e => reachyPost(`/api/reachy/volume/mic/${e.target.value}`));
  $("reachySpeakButton").addEventListener("click", () => {
    const text = $("reachySpeakText").value.trim();
    if (text) reachyPost("/api/reachy/speak", {text});
  });
  $("reachySpeakAnswer").addEventListener("click", () => {
    if (!state.completed || $("answer").classList.contains("streaming")) {
      error("No completed caption to speak yet - wait for one to finish.");
      return;
    }
    const text = $("answer").textContent.trim();
    if (text) reachyPost("/api/reachy/speak", {text});
  });
  pollReachyControlState();
  setInterval(pollReachyControlState, 5000);

  // Real elapsed time against the server's own switch_progress.timeout - never a fabricated
  // percentage. started_at/timeout come from whichever switcher is active (ServiceEngineSwitcher
  // or the legacy symlink EngineSwitcher, both now report it - see /api/engines in serve_ui.py).
  // Ticks on a local timer between the 5 s /api/engines polls so the bar moves smoothly; capped
  // short of 100% until the switch actually reports done, since "done" is a real event, not a
  // time estimate.
  function updateSwitchProgress() {
    const bar = $("engineSwitchProgress"), fill = $("engineSwitchProgressFill");
    const active = switchingEngine();
    bar.hidden = !active;
    if (!active) {
      if (switchProgressTimer) { clearInterval(switchProgressTimer); switchProgressTimer = null; }
      fill.style.width = "0%";
      return;
    }
    const progress = engineData?.switch_progress;
    const pct = progress && Number.isFinite(progress.started_at) && progress.timeout > 0
      ? Math.max(2, Math.min(96, (Date.now() / 1000 - progress.started_at) / progress.timeout * 100))
      : 2;
    fill.style.width = `${pct}%`;
    bar.setAttribute("aria-valuenow", String(Math.round(pct)));
    if (!switchProgressTimer) switchProgressTimer = setInterval(updateSwitchProgress, 300);
  }
  // One health/engine poll owns the complete model snapshot. Switches retire it
  // and refresh every endpoint on success and rollback; camera/robot preview stays.
  function renderEngines() {
    if (!engineData) return;
    $("engineSwitchRow").hidden = false;
    $("engineSwitchHint").hidden = false;
    renderEngineChoices($("modelButtons"), engineData, switchingEngine(), switchToEngine);
    $("engineSwitchStatus").textContent = switchingEngine()
      ? "Switching model… Preview stays connected; inference is paused."
      : `Current model: ${engineData.active?.name || "Unavailable"}`;
    $("engineSwitchHint").textContent = engineData.configured
      ? "Only one model is resident at a time. Switching pauses inference while the new model loads; a failed switch attempts to restore the previous model."
      : "Model switching is not configured on this device.";
    updateSwitchProgress();
  }
  async function switchToEngine(id) {
    const choice = engineData && engineChoices(engineData).find(engine => engine.id === id);
    if (switchingEngine() || !choice?.available || id === engineData.active?.id) return;
    try {
      await runEngineSwitch(engineRequests, {
        before() {
          engineSwitching = true; state.checking = null;
          retireEngineAnswer("Switching model · inference paused"); error();
          $("backendStatus").textContent = "Switching model…";
          $("backendStatus").className = "badge unavailable";
          renderEngines(); controls();
        },
        async request() {
          const credentials = await loadAccess();
          const response = await fetch(`/api/engines/${encodeURIComponent(id)}`, {
            method: "POST", credentials: "same-origin", headers: {"X-Reachy-Token": credentials.reachy_token},
            signal: AbortSignal.timeout(420000)
          });
          const body = await response.json().catch(() => null);
          if (response.status === 401) access = null;
          if (!response.ok || body?.ok === false) throw new Error(body?.error?.message || body?.message || `Model switch failed (HTTP ${response.status}).`);
        },
        after() {
          engineSwitching = false;
          resetTimingGroup("Model switch finished · waiting for a new answer.");
        },
        refresh: checkBackend,
      });
    } catch (err) {
      error(`${err.message} Check the current model status before retrying.`);
    } finally { renderEngines(); controls(); }
  }

  // Services panel: what's running, its RAM and on-disk size, and SD card vs NVMe. Polled
  // independently of engine switching - a service can be up or down regardless of which model is
  // currently active.
  async function refreshServices() {
    try {
      const response = await fetch("/api/services", {cache: "no-store"});
      const data = await response.json();
      const fmtMb = mb => mb === null || mb === undefined ? "—" : mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
      // A switchable model's own service is stopped whenever it is not the currently selected
      // one - expected, not a fault. "active" (present only for entries backed by the engine
      // registry - see service_list() in serve_ui.py) tells the two apart: an unselected model
      // reads "not selected" rather than the alarming bare "stopped" it used to, and the one real
      // fault state - selected but its process is not actually up - gets its own visible class.
      $("servicesList").innerHTML = (data.services || []).map(s => {
        const managed = "active" in s;
        const fault = managed && s.active && !s.running;
        const rowClass = fault ? "fault" : !managed ? (s.running ? "running" : "stopped")
          : s.active ? "running" : "standby";
        const label = fault ? "active · not responding"
          : !managed ? (s.running ? "running" : "stopped")
          : s.active ? "active · running" : (s.running ? "running (not selected)" : "not selected");
        return `
        <li class="service-row ${rowClass}">
          <span class="service-name"><span class="service-dot"></span>${s.name}</span>
          <span class="service-detail">${label} · RAM ${fmtMb(s.memory_mb)} · disk ${fmtMb(s.storage_mb)} (${s.disk})</span>
        </li>`;
      }).join("") || `<li class="hint">No services configured.</li>`;
    } catch (_) { /* keep last known list on a transient poll failure */ }
  }
  refreshServices();
  setInterval(refreshServices, 5000);
}
