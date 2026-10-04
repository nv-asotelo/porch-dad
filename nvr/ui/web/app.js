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

// This repo's own quick presets, after the Live VLM WebUI ones: name the Pokémon and locate it.
// Cosmos3-Edge answers them with JSON (parseGrounding) and per-token logprobs (nameProbability); a
// classifier answers them with its top species and a location derived from its saliency map
// (saliencyLocation). "locate" says which mark the preset draws.
const POKEMON_PRESETS = [
  {label: "🔎 Name the Pokémon · box", locate: "box",
    prompt: "Which Pokémon is this? Reply with only a JSON object with \"name\" (its English name) and \"bbox_2d\" (its bounding box [x1, y1, x2, y2])."},
  {label: "🔎 Name the Pokémon · point", locate: "point",
    prompt: "Which Pokémon is this? Reply with only a JSON object with \"name\" (its English name) and \"point_2d\" (a point [x, y] on it)."},
];
// Per generated token, the shim's top alternatives with their logprobs (log-softmax of the raw logits).
// The presets decode greedily, so the chosen token is the first; 5 keeps it in the list through ties.
const POKEMON_TOP_LOGPROBS = 5;

function pokemonPreset(prompt) {
  return POKEMON_PRESETS.find(preset => preset.prompt === prompt) ?? null;
}

// Where a classifier's evidence is: ONE box [x1, y1, x2, y2] and ONE point [x, y], normalized 0-1
// over the picture it was sent, from its saliency grid ({w, h, cells}, row-major, scaled to its peak),
// or null when the grid is all zero or flat. Deterministic:
//  1. peak = first maximum cell; threshold T = base + threshold x (max - base), base 0 ("max") or the
//     smallest cell ("range", for attention rollout, which never reaches 0).
//  2. Bilinear upsampling by an odd factor (align_corners=false, edge-clamped: what the overlay's
//     smoothed drawImage shows), so one sample sits on each cell centre.
//  3. Keep samples >= T: the 4- or 8-connected component holding the peak ("peak"), the largest one
//     ("largest"), or all of them ("all").
//  4. box = their extent, grown or shrunk by `pad` cells per side; point = the peak cell's centre, or
//     the centroid weighted by value ("centroid") or by how far each exceeds T ("excess").
// An estimate of where the evidence is, not a detection. Tuned on the 189-image sample set against
// Cosmos3-Edge's boxes (a pseudo-reference): Skshmjn median IoU 0.74, point inside the Cosmos box 99%;
// Bierny (a coarse 7x7 Grad-CAM) median IoU 0.41, point inside 76%.
const SALIENCY_LOCATION_PARAMS = Object.freeze({
  skshmjn: Object.freeze({threshold: 0.35, relative: "range", upsample: 9, connectivity: 4,
    component: "peak", pad: -0.25, point: "excess"}),
  bierny: Object.freeze({threshold: 0.05, relative: "max", upsample: 9, connectivity: 4,
    component: "all", pad: -0.5, point: "centroid"}),
});
const SALIENCY_LOCATION_DEFAULT = Object.freeze({threshold: 0.5, relative: "max", upsample: 9,
  connectivity: 4, component: "peak", pad: 0, point: "centroid"});
const NEIGHBOURS = {
  4: [[0, -1], [-1, 0], [1, 0], [0, 1]],
  8: [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
};

function checkParams(p) {
  const ok = p && typeof p === "object" &&
    typeof p.threshold === "number" && p.threshold > 0 && p.threshold < 1 &&
    (p.relative === "max" || p.relative === "range") &&
    Number.isInteger(p.upsample) && p.upsample >= 1 && p.upsample <= 15 && p.upsample % 2 === 1 &&
    (p.connectivity === 4 || p.connectivity === 8) &&
    (p.component === "peak" || p.component === "largest" || p.component === "all") &&
    typeof p.pad === "number" && Number.isFinite(p.pad) && Math.abs(p.pad) <= 2 &&
    (p.point === "peak" || p.point === "centroid" || p.point === "excess");
  if (!ok) throw new TypeError("saliencyLocation: invalid params");
}

// Sample k of an axis upsampled S times: the two cells it blends and the second one's weight (an
// integer numerator, so this matches the Python reference bit for bit).
function axisSample(k, size, S) {
  const num = 2 * k + 1 - S;
  let i0 = Math.floor(num / (2 * S));
  let f = (num - 2 * S * i0) / (2 * S);
  if (i0 < 0) { i0 = 0; f = 0; }
  if (i0 >= size - 1) { i0 = size - 1; f = 0; }
  return [i0, Math.min(i0 + 1, size - 1), f];
}

function upsampleField(cells, w, h, S) {
  const W = w * S, H = h * S, field = new Float64Array(W * H);
  const xs = [], ys = [];
  for (let k = 0; k < W; k++) xs.push(axisSample(k, w, S));
  for (let k = 0; k < H; k++) ys.push(axisSample(k, h, S));
  for (let yi = 0; yi < H; yi++) {
    const [y0, y1, fy] = ys[yi];
    for (let xi = 0; xi < W; xi++) {
      const [x0, x1, fx] = xs[xi];
      const a = cells[y0 * w + x0], b = cells[y0 * w + x1], c = cells[y1 * w + x0], d = cells[y1 * w + x1];
      field[yi * W + xi] = (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d);
    }
  }
  return field;
}

// Label every connected set of samples >= T, starting scans in row-major order. {label (-1 outside), sizes}.
function labelComponents(field, W, H, T, connectivity) {
  const n = W * H, label = new Int32Array(n).fill(-1), stack = new Int32Array(n), sizes = [];
  const offs = NEIGHBOURS[connectivity];
  for (let start = 0; start < n; start++) {
    if (!(field[start] >= T) || label[start] >= 0) continue;
    const lab = sizes.length;
    let top = 0, size = 0;
    label[start] = lab; stack[top++] = start;
    while (top > 0) {
      const i = stack[--top];
      size += 1;
      const x = i % W, y = (i - x) / W;
      for (const [dx, dy] of offs) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
        const j = ny * W + nx;
        if (field[j] >= T && label[j] < 0) { label[j] = lab; stack[top++] = j; }
      }
    }
    sizes.push(size);
  }
  return {label, sizes};
}

function saliencyLocation(saliency, params = SALIENCY_LOCATION_DEFAULT) {
  checkParams(params);
  const {w, h, cells} = saliency || {};
  if (!Number.isInteger(w) || !Number.isInteger(h) || w < 1 || h < 1 || w > 64 || h > 64 ||
      !cells || cells.length !== w * h) throw new TypeError("saliencyLocation: invalid saliency grid");
  const n = w * h;
  for (let i = 0; i < n; i++) {
    if (typeof cells[i] !== "number" || !Number.isFinite(cells[i])) throw new TypeError("saliencyLocation: invalid saliency grid");
  }
  let peak = 0, vmax = cells[0], vmin = cells[0];
  for (let i = 1; i < n; i++) {
    const v = cells[i];
    if (v > vmax) { vmax = v; peak = i; }
    if (v < vmin) vmin = v;
  }
  if (!(vmax > 0) || vmax === vmin) return null;
  const base = params.relative === "range" ? vmin : 0;
  const T = base + params.threshold * (vmax - base);
  const S = params.upsample, W = w * S, H = h * S;
  const field = S > 1 ? upsampleField(cells, w, h, S) : Float64Array.from(cells);
  const seed = ((peak - (peak % w)) / w * S + (S - 1) / 2) * W + (peak % w) * S + (S - 1) / 2;
  const {label, sizes} = labelComponents(field, W, H, T, params.connectivity);
  if (label[seed] < 0) return null;
  let keep = null;
  if (params.component === "peak") keep = label[seed];
  else if (params.component === "largest") {
    keep = label[seed];
    for (let lab = 0; lab < sizes.length; lab++) if (sizes[lab] > sizes[keep]) keep = lab;
  }
  let xmin = W, ymin = H, xmax = -1, ymax = -1, sv = 0, sx = 0, sy = 0;
  const excess = params.point === "excess";
  for (let i = 0; i < W * H; i++) {
    if (keep === null ? label[i] < 0 : label[i] !== keep) continue;
    const x = i % W, y = (i - x) / W;
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
    const wt = excess ? field[i] - T : field[i];
    sv += wt; sx += wt * (x + 0.5); sy += wt * (y + 0.5);
  }
  const pad = params.pad, clamp = v => Math.min(1, Math.max(0, v));
  const box = [clamp((xmin / S - pad) / w), clamp((ymin / S - pad) / h),
    clamp(((xmax + 1) / S + pad) / w), clamp(((ymax + 1) / S + pad) / h)];
  const point = params.point === "peak" || !(sv > 0)
    ? [((peak % w) + 0.5) / w, ((peak - (peak % w)) / w + 0.5) / h]
    : [sx / sv / W, sy / sv / H];
  return {box, point};
}

// A Name-the-Pokémon answer from Cosmos3-Edge: each grounding item with the probability of the name
// it wrote, a caption, and overlay marks. Null when the answer holds no grounding JSON.
function readPokemonAnswer(text, logprobs) {
  const items = parseGrounding(text);
  if (!items) return null;
  const percent = p => p === null ? "" : ` ${percentText(p)}`;
  const read = items.map(item => ({...item, probability: item.name ? nameProbability(logprobs, item.name) : null}));
  return {items: read,
    caption: read.map(item => `${item.name ?? "No name given"}${item.probability === null ? "" : ` ·${percent(item.probability)}`}`).join("\n"),
    marks: read.filter(item => item.box || item.point).map(item => ({box: item.box, point: item.point,
      label: `${item.name ?? "?"}${percent(item.probability)}`, derived: false}))};
}

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
  if (!policy) return null;
  if (typeof policy.prompt !== "string" || !policy.prompt.trim() ||
      typeof policy.max_tokens !== "number" || typeof policy.temperature !== "number" ||
      typeof policy.image_tokens !== "number" || policy.stream !== false) {
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
    ...(Array.isArray(choice?.logprobs?.content) ? {logprobs: choice.logprobs.content} : {}),
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
    // The sampler runs every 250ms. A longer gap does not imply observed continuity.
    if (!current || point.breakBefore || point.at - previous.at > 600 || point.at <= previous.at) {
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
      status(partial ? "Partial metrics · some readings unavailable" : "Live · updates 4 times a second", partial ? "partial" : "live");
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
    tick(); interval = setInterval(tick, 250);
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

// Classifiers ("kind": "classifier" models) answer with a ranking and an optional saliency grid,
// not text. Species are canonical lowercase names ("mr-mime"); null is a label the model has that
// maps to no species, shown by its raw label.
function speciesName(species, label = "") {
  if (typeof species !== "string" || !species) return label || "Unknown";
  return species.split("-").map(part => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function readClassification(value) {
  if (!value || typeof value !== "object" || !Array.isArray(value.topk) || !value.topk.length) {
    throw new Error("The classifier sent no ranking.");
  }
  const score = x => {
    if (typeof x !== "number" || !Number.isFinite(x) || x < 0 || x > 1) throw new Error("The classifier sent an invalid score.");
    return x;
  };
  const topk = value.topk.map(item => ({species: typeof item?.species === "string" ? item.species : null,
    label: typeof item?.label === "string" ? item.label : "", score: score(item?.score)}));
  let saliency = null;
  if (value.saliency) {
    const {w, h, cells, method} = value.saliency;
    if (!Number.isInteger(w) || !Number.isInteger(h) || w < 1 || h < 1 || w > 64 || h > 64 ||
        !Array.isArray(cells) || cells.length !== w * h) throw new Error("The classifier sent an invalid saliency map.");
    saliency = {w, h, cells: cells.map(score), method: typeof method === "string" ? method : ""};
  }
  const inference = value.timing_ms?.inference;
  return {species: topk[0].species, label: topk[0].label, score: topk[0].score, topk, saliency,
    boxes: Array.isArray(value.boxes) ? value.boxes : [],
    inferenceMs: typeof inference === "number" && Number.isFinite(inference) ? inference : null};
}

function percentText(p) {
  return `${p >= 0.1 ? Math.round(p * 100) : (p * 100).toFixed(1)}%`;
}

function describeClassification(result) {
  const percent = percentText;
  const [best, ...rest] = result.topk;
  const runnersUp = rest.map((r, i) => `${i + 2}. ${speciesName(r.species, r.label)} ${percent(r.score)}`).join(" · ");
  return `${speciesName(best.species, best.label)} · ${percent(best.score)}${runnersUp ? `\n${runnersUp}` : ""}`;
}

// Where a saliency grid lands in the preview. The grid covers the whole image the classifier was
// sent: the capture canvas, with the source drawn at (x, y, w, h) inside it. The preview shows the
// source with object-fit: contain in a boxWidth × boxHeight box. Returns that visible picture
// (clip) and the rectangle the full grid stretches over (grid), both in box pixels.
function overlayPlacement(sent, natural, boxWidth, boxHeight) {
  const scale = Math.min(boxWidth / natural.width, boxHeight / natural.height);
  const clip = {width: natural.width * scale, height: natural.height * scale};
  clip.x = (boxWidth - clip.width) / 2; clip.y = (boxHeight - clip.height) / 2;
  const perX = clip.width / sent.w, perY = clip.height / sent.h;
  return {clip, grid: {x: clip.x - sent.x * perX, y: clip.y - sent.y * perY,
    width: sent.canvasWidth * perX, height: sent.canvasHeight * perY}};
}

// A sequential dark-to-bright ramp (inferno-like), so more evidence always reads brighter; the
// overlay's alpha also rises with the value, so low-evidence areas leave the picture visible.
const HEAT_STOPS = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];
function heatColor(value) {
  const v = Math.min(1, Math.max(0, value)) * (HEAT_STOPS.length - 1);
  const i = Math.min(HEAT_STOPS.length - 2, Math.floor(v)), t = v - i;
  return HEAT_STOPS[i].map((c, k) => Math.round(c + (HEAT_STOPS[i + 1][k] - c) * t));
}

// A species name as a canonical identifier, the way the sample set and the classifiers spell
// species: "Mr. Mime" -> "mr-mime", "Nidoran♀" -> "nidoran-f", "Farfetch'd" -> "farfetchd".
function speciesKey(name) {
  if (typeof name !== "string") return null;
  const key = name.trim().replace(/♀/g, "-f").replace(/♂/g, "-m").normalize("NFKD")
    .replace(/[̀-ͯ]/g, "").toLowerCase().replace(/['’.]/g, "").replace(/:/g, "-")
    .replace(/[\s_]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
  return key || null;
}

// 2D grounding in a Name-the-Pokémon answer: the first name and the first box or point Cosmos3-Edge
// wrote, as [{name, box, point}], or null. A plain JSON.parse fails on about 1 answer in 20 (extra
// "]]", a second object after the first, a cut-off answer) and keeps the LAST of a duplicated key,
// where the model's first is the one it meant. So this reads values directly: the first "name" (or
// "label", "species") string, and the first array of 4 (box) or 2 (point) numbers after the first
// "bbox_2d" (or "point_2d") key - inside nested objects too; "bbox_2d": null means no box. Fences,
// lists and pretty-printing make no difference. Coordinates are normalized 0-1000 from the top-left
// of the picture sent, whatever the prompt asks for (measured), and come back as 0-1 fractions.
function parseGrounding(text) {
  if (typeof text !== "string") return null;
  const body = text.replace(/```[A-Za-z]*/g, "");
  let name = null;
  const nameMatch = /"(?:name|label|species)"\s*:\s*"((?:[^"\\]|\\.)*)"/i.exec(body);
  if (nameMatch) {
    try { name = JSON.parse(`"${nameMatch[1]}"`); } catch (_) { name = nameMatch[1]; }
    name = name.trim().replace(/^[\s"'`]+|[\s"'`.]+$/g, "") || null;
  }
  const number = "\\s*(-?\\d+(?:\\.\\d+)?)\\s*";
  const coords = (keys, n) => {
    const key = new RegExp(`"(?:${keys})"\\s*:\\s*(null)?`, "i").exec(body);
    if (!key || key[1]) return null;
    const array = new RegExp(`\\[${Array(n).fill(number).join(",")}\\]`).exec(body.slice(key.index + key[0].length));
    return array ? array.slice(1).map(c => Math.min(1, Math.max(0, Number(c) / 1000))) : null;
  };
  let box = coords("bbox_2d|bbox|box", 4);
  if (box && (box[2] <= box[0] || box[3] <= box[1])) box = null;
  const point = coords("point_2d|point", 2);
  return name || box || point ? [{name, box, point}] : null;
}

// How likely the model thought the name it wrote was: the product of its tokens' probabilities,
// from OpenAI-style per-token logprobs ({token, logprob, bytes}). The name's span is found in the
// text the tokens spell; a token that straddles the span's edge (one carrying the opening quote,
// say) counts whole, so this errs low. Null when the logprobs do not spell out the name.
function nameProbability(content, name) {
  if (!Array.isArray(content) || !content.length || typeof name !== "string" || !name) return null;
  // A chosen token outside the returned top-K comes back with bytes and logprob null (sampling).
  if (content.some(entry => entry?.bytes === null || entry?.logprob === null)) return null;
  const encoder = new TextEncoder();
  const pieces = content.map(entry => Array.isArray(entry?.bytes) ? Uint8Array.from(entry.bytes)
    : encoder.encode(typeof entry?.token === "string" ? entry.token : ""));
  const all = new Uint8Array(pieces.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  const spans = pieces.map(p => { all.set(p, offset); const span = [offset, offset + p.length]; offset += p.length; return span; });
  const text = new TextDecoder().decode(all);
  const found = text.indexOf(name);
  if (found < 0) return null;
  const from = encoder.encode(text.slice(0, found)).length, to = from + encoder.encode(name).length;
  let logprob = 0, used = 0;
  content.forEach((entry, i) => {
    if (spans[i][1] > from && spans[i][0] < to && typeof entry?.logprob === "number" && Number.isFinite(entry.logprob)) {
      logprob += entry.logprob; used += 1;
    }
  });
  return used ? Math.exp(logprob) : null;
}

// A Name-the-Pokémon answer as a sample result, in the classifiers' shape: the first name it gave as
// the ranking's only entry, scored by that name's probability (null without logprobs).
function cosmosResult(named) {
  const item = named?.items.find(entry => entry.name) ?? null;
  const entry = {species: item ? speciesKey(item.name) : null, label: item?.name ?? "", score: item?.probability ?? null};
  return {...entry, topk: [entry], cosmos: true};
}

// Score of a sample run: rows of {truth, covered (false when the model cannot name that species
// at all), ranked: [species, ...] best first}.
function scoreSamples(rows) {
  const score = {n: 0, top1: 0, top5: 0, coveredN: 0, coveredTop1: 0};
  for (const row of rows) {
    const top1 = row.ranked[0] === row.truth, top5 = row.ranked.slice(0, 5).includes(row.truth);
    score.n += 1; score.top1 += top1; score.top5 += top5;
    if (row.covered !== false) { score.coveredN += 1; score.coveredTop1 += top1; }
  }
  return score;
}

if (typeof module !== "undefined") module.exports = {SSEParser, readCompletionEvent, CAPTURE_PRESETS, FrameCadence, PROMPT_PRESETS, LatencySummary, appendTelemetrySample, telemetrySegments, ADVANCED_DEFAULTS, validateAdvancedSettings, readServerInferenceMs, readServerFirstTextMs, describeReachyHealth, readRelayStream, readRelaySlots, skippedSampleReason, REACHY_SAMPLE_MS, EngineRequestScope, EnginePolicySettings, enginePolicy, engineChoices, renderEngineChoices, runEngineSwitch, speciesName, readClassification, describeClassification, overlayPlacement, heatColor, scoreSamples, speciesKey, parseGrounding, nameProbability, POKEMON_PRESETS, pokemonPreset, saliencyLocation, SALIENCY_LOCATION_PARAMS, readPokemonAnswer, cosmosResult, percentText};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const state = {ready: false, model: "", running: false, busy: false, media: null,
    abort: null, captureAt: null, completed: 0, imageURL: null, checking: false, cameraGeneration: 0,
    preset: "lightweight", frameCallback: null, sampled: 0, skipped: 0,
    liveStreaming: true, activeTrigger: null, engineId: null, timingGroup: 0,
    advanced: {...ADVANCED_DEFAULTS}, advancedValid: true, source: "camera",
    policy: null, activeEngineId: null, remoteSwitching: false,
    kind: "cosmos", overlays: true, sampleId: null};
  const engineRequests = new EngineRequestScope(), policySettings = new EnginePolicySettings();
  let engineSwitching = false, engineData = null, switchProgressTimer = null;
  // The latest classifier saliency and where it was captured, redrawn over the preview on resize.
  let overlay = null;
  // The sample set (/api/samples): its images, coverage for the selected classifier, and that
  // classifier's results so far. A run is cancelled by Stop, a model switch or a new run.
  const samples = {list: [], byId: new Map(), results: new Map(), model: null, run: null};
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
  // #reachyImage is bound to the long-running /reachy/mjpeg multipart stream for the live
  // preview a human watches, and that keeps painting fine - but the browser's decoded bitmap
  // backing drawImage() on that element can stop advancing even while fresh bytes keep
  // arriving over the wire (verified server-side: every layer up to and including a direct
  // fetch of /reachy/mjpeg delivers genuinely new frames continuously; only what an <img>
  // bound to that long-lived stream hands to canvas can go stale). Inference correctness
  // cannot depend on that painting behaviour, so each Reachy capture instead does its own
  // one-shot fetch of /reachy/still.jpg - a plain request/response with no long-lived
  // decode state to go stale - and draws that instead of the preview element.
  async function fetchReachyStill() {
    if (!access) throw new Error("Reachy access token not available yet.");
    const response = await fetch(reachyURL("/reachy/still.jpg"), {cache: "no-store"});
    if (!response.ok) throw new Error("Could not fetch a fresh frame from Reachy.");
    const blob = await response.blob();
    return createImageBitmap(blob);
  }
  // ?source=reachy (a bookmarkable link) selects the robot once inference can run.
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
  const pokemonGroup = document.createElement("optgroup");
  pokemonGroup.label = "Pokémon · Cosmos3-Edge and the classifiers";
  for (const preset of POKEMON_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.prompt; option.textContent = preset.label;
    pokemonGroup.append(option);
  }
  $("promptPreset").append(pokemonGroup);
  function matchPromptPreset() {
    $("promptPreset").value = PROMPT_PRESETS.some(preset => preset.prompt === $("prompt").value)
      ? $("prompt").value : "";
  }
  $("promptPreset").addEventListener("change", () => {
    if ($("promptPreset").value) $("prompt").value = $("promptPreset").value;
    if (pokemonPreset($("prompt").value) && Number($("maxTokens").value) < 96) $("maxTokens").value = "96";
    resetSampleResults();
    controls();
  });
  $("prompt").addEventListener("input", () => { matchPromptPreset(); resetSampleResults(); controls(); });
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
    for (const id of ["prompt", "maxTokens", "imageTokenPreset", "customImageTokens", "topP"]) {
      $(id).disabled = Boolean(state.policy) || state.kind === "classifier" || switchingEngine();
    }
    // A classifier has no prompt, but the Name-the-Pokémon presets say what it returns: its
    // species, plus a box or point derived from its saliency map.
    $("promptPreset").disabled = Boolean(state.policy) || switchingEngine();
    for (const option of $("promptPreset").querySelectorAll("option")) {
      option.disabled = state.kind === "classifier" && option.value !== "" && !pokemonPreset(option.value);
    }
    for (const id of ["lightweightPreset", "liveVlmPreset"]) $(id).disabled = Boolean(state.policy) || switchingEngine();
    const pokemon = pokemonPreset($("prompt").value.trim());
    $("overlayToggleButton").hidden = state.kind !== "classifier" && !pokemon;
    $("overlayToggleButton").setAttribute("aria-pressed", String(state.overlays));
    $("overlayToggleButton").textContent = `Saliency overlay: ${state.overlays ? "On" : "Off"}`;
    $("sampleRunAll").disabled = !(state.kind === "classifier" || pokemon) || !state.ready || state.busy ||
      switchingEngine() || !samples.list.length;
    $("sampleRunAll").title = state.kind === "classifier" || pokemon ? ""
      : "Choose a Name the Pokémon quick preset to score Cosmos3-Edge on the sample set";
    $("sampleStop").hidden = !samples.run;
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
    // Saliency and sample results belong to the model that produced them.
    overlay = null; drawOverlay();
    if (samples.run) samples.run.cancelled = true;
    samples.results.clear(); samples.model = null; renderSampleTiles();
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
      ? "This model uses a fixed prompt and settings; the controls above are disabled while it is active." : "";
    $("firstTextLabel").textContent = policy ? "Token timing unavailable" : "First visible token";
    $("serverTtftHelp").textContent = policy
      ? "This model returns one complete answer. Native TTFT and token speed are not reported; Round trip is browser time to the complete response."
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
      const changed = (active?.id ?? null) !== state.activeEngineId;
      state.activeEngineId = active?.id ?? null;
      state.kind = active?.kind === "classifier" ? "classifier" : "cosmos";
      if (changed) loadSamples(state.kind === "classifier" ? active.model_id : null);
      if (state.kind === "classifier") {
        // Served by the classifier service, not the shim: the shim's health does not gate it.
        if (active.available === false) throw new Error(active.reason || "The selected classifier is not available");
        const restored = policySettings.apply(null, modelSettings());
        state.ready = true; state.model = active.model_id;
        $("backendStatus").textContent = "Classifier ready";
        $("backendStatus").className = "badge ready";
        $("modelName").textContent = active.name;
        $("staticClocksValue").textContent = "Not applicable"; $("encoderCacheValue").textContent = "Not applicable";
        $("runtimeStatus").textContent = "A classifier has no engine settings: prompt, token limits and sampling do not apply to it.";
        applyModelPolicy(null, restored);
        $("modelPolicyStatus").hidden = false;
        $("modelPolicyStatus").textContent = "A Pokémon classifier is selected. It takes no prompt and has no token or sampling settings: each image gets its most likely species, the next four, and a saliency overlay.";
        $("firstTextLabel").textContent = "Token timing unavailable";
        $("serverTtftHelp").textContent = "A classifier answers in one step: Latency is its inference time on the server, and TTFT does not apply.";
      } else {
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
            ? "This model uses its own fixed request policy. Device clocks and cache sizes are not reported by this endpoint."
            : "Engine settings unavailable. Input controls keep their current values; clocks and cache cannot be verified.";
        }
        applyModelPolicy(policy, restored);
      }
    } catch (err) {
      if (!engineRequests.current(ticket) || state.checking !== ticket) return;
      state.ready = false;
      $("backendStatus").textContent = state.remoteSwitching ? "Switching model…" : "Backend not ready";
      $("backendStatus").className = "badge unavailable";
      $("modelName").textContent = state.kind === "classifier" ? "Waiting for the classifier service" : "Waiting for local TensorRT-Edge-LLM";
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
    const width = source.videoWidth || source.naturalWidth || source.width;
    const height = source.videoHeight || source.naturalHeight || source.height;
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
    const x = Math.floor((canvas.width - imageWidth) / 2), y = Math.floor((canvas.height - imageHeight) / 2);
    context.drawImage(source, x, y, imageWidth, imageHeight);
    const capturedAt = performance.now();
    const url = canvas.toDataURL("image/jpeg", preset.jpegQuality);
    if (url.length > 1900000) throw new Error("This full-size frame exceeds the image request limit. Choose a smaller source or the Lightweight preset.");
    $("captureStatus").textContent = `Sent ${canvas.width}×${canvas.height} · ${state.preset === "live-vlm" ? "Live VLM WebUI" : "Lightweight"}`;
    // Where the picture sits in what was sent, for drawing a classifier's saliency back over it.
    return {url, capturedAt, natural: {width, height},
      sent: {canvasWidth: canvas.width, canvasHeight: canvas.height, x, y, w: imageWidth, h: imageHeight}};
  }
  async function analyze(source, trigger = "manual") {
    if (state.busy || !state.ready || !state.advancedValid || switchingEngine()) return;
    if (state.kind === "classifier") {
      const sample = source === $("uploadedImage") ? state.sampleId : null;
      await classify(source, trigger, sample);
      return;
    }
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
    const pokemon = policy ? null : pokemonPreset(prompt);
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
    const temperature = policy?.temperature ?? (pokemon ? 0 : CAPTURE_PRESETS[state.preset].temperature);
    const ticket = engineRequests.begin(), controller = ticket.controller;
    state.busy = true; state.abort = controller; state.activeTrigger = trigger; controls(); error();
    $("ttft").textContent = "—"; $("totalTime").textContent = "—";
    $("runStatus").textContent = "Reading frame…";
    let reader, counted = false;
    // Nothing answered at all: the network failed, not this request (see skippedSampleReason).
    const network = err => { if (err?.name === "TypeError") err.network = true; throw err; };
    try {
      const captureSource = source === $("reachyImage") ? await fetchReachyStill() : source;
      const image = capture(captureSource);
      state.captureAt = image.capturedAt;
      counted = state.running && source === liveSource();
      if (counted) state.sampled += 1;
      const started = performance.now();
      let firstToken = null, done = false, output = "", finishReason = null, serverMetrics = null;
      const tokenLogprobs = [];
      const response = await fetch("/v1/chat/completions", {
        method: "POST", headers: {"Content-Type": "application/json"}, signal: controller.signal,
        body: JSON.stringify({model: state.model, stream: true, temperature,
          top_p: advanced.topP, max_image_tokens_per_image: advanced.imageTokens,
          stream_options: {include_usage: true},
          ...(pokemon ? {logprobs: true, top_logprobs: POKEMON_TOP_LOGPROBS} : {}),
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
        if (completion.logprobs) tokenLogprobs.push(...completion.logprobs);
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
      // A Name-the-Pokémon answer is JSON: show the name and how likely the model found it, draw
      // its box or point, and keep the raw answer one hover away.
      let spoken = output;
      const named = pokemon ? readPokemonAnswer(output, tokenLogprobs) : null;
      $("answer").title = named ? output : "";
      if (named) {
        $("answer").textContent = named.caption;
        spoken = named.items.map(item => item.name).filter(Boolean).join(", ") || output;
        overlay = named.marks.length ? {natural: image.natural, sent: image.sent, marks: named.marks} : null;
        drawOverlay();
        const sample = source === $("uploadedImage") ? state.sampleId : null;
        const key = `${state.model}|${prompt}`;
        if (sample && samples.byId.has(sample) && (samples.model === null || samples.model === key)) {
          samples.model = key; samples.results.set(sample, cosmosResult(named)); renderSampleTiles();
        }
      }
      // speakText() itself no-ops while a previous utterance is still in flight (see its own
      // "speaking" guard below), which is exactly the behaviour wanted here: a caption that lands
      // while Reachy is still speaking the last one is skipped, not queued, so auto-speak tracks
      // captioning as closely as the selected rate allows without ever building a backlog.
      if (autoSpeak) speakText(spoken);
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
  // One frame, or one sample by id, through the selected classifier: the same request rules as
  // analyze() (one at a time, a model switch retires it, a bad live sample is skipped), but one
  // complete JSON answer. A sample is sent by id so the server classifies its own copy of the file,
  // exactly as a sample-set run does, rather than a re-encoded capture of it.
  async function classify(source, trigger = "manual", sampleId = null) {
    if (source === $("reachyImage") && !reachyFrameReady()) { renderReachySampling(); controls(); return null; }
    // A Name-the-Pokémon preset needs the saliency map even with the overlay off: it is where the
    // box or point comes from.
    const pokemon = pokemonPreset($("prompt").value.trim());
    const saliency = state.overlays || Boolean(pokemon);
    const keepSource = trigger === "live" && state.source === "reachy";
    const timingGroup = state.timingGroup;
    const ticket = engineRequests.begin(), controller = ticket.controller;
    state.busy = true; state.abort = controller; state.activeTrigger = trigger; controls(); error();
    $("ttft").textContent = "—"; $("totalTime").textContent = "—";
    $("runStatus").textContent = "Reading frame…";
    let counted = false;
    const network = err => { if (err?.name === "TypeError") err.network = true; throw err; };
    try {
      let request, placement;
      if (sampleId) {
        const shown = $("uploadedImage"), width = shown.naturalWidth, height = shown.naturalHeight;
        placement = {natural: {width, height}, sent: {canvasWidth: width, canvasHeight: height, x: 0, y: 0, w: width, h: height}};
        request = {model: state.model, sample: sampleId, saliency, topk: 5};
        state.captureAt = performance.now();
        $("captureStatus").textContent = `Sample ${width}×${height} · the server's copy`;
      } else {
        const image = capture(source === $("reachyImage") ? await fetchReachyStill() : source);
        state.captureAt = image.capturedAt; placement = {natural: image.natural, sent: image.sent};
        counted = state.running && source === liveSource();
        if (counted) state.sampled += 1;
        request = {model: state.model, image: image.url, saliency, topk: 5};
      }
      $("runStatus").textContent = "Classifying…";
      const started = performance.now();
      const response = await fetch("/api/classify", {
        method: "POST", headers: {"Content-Type": "application/json"}, signal: controller.signal,
        body: JSON.stringify(request)
      }).catch(network);
      const body = await response.json().catch(() => null);
      controller.signal.throwIfAborted();
      if (state.abort !== controller || !engineRequests.current(ticket)) return null;
      if (!response.ok) {
        throw Object.assign(new Error(body?.error?.message || `Classifier returned HTTP ${response.status}.`),
          {status: response.status});
      }
      const result = readClassification(body);
      $("totalTime").textContent = duration(performance.now() - started);
      $("answer").textContent = describeClassification(result);
      if (timingGroup === state.timingGroup) {
        serverFirstTextMs = null;
        if (result.inferenceMs === null) latency.lastMs = null; else latency.add(result.inferenceMs);
        $("timingStatus").textContent = "Classifier inference on the server · excludes image decoding and preprocessing.";
        renderLatency();
      }
      const location = pokemon && result.saliency
        ? (() => { try { return saliencyLocation(result.saliency, SALIENCY_LOCATION_PARAMS[state.model] ?? SALIENCY_LOCATION_DEFAULT); } catch (_) { return null; } })()
        : null;
      const marks = location ? [{box: pokemon.locate === "box" ? location.box : null,
        point: pokemon.locate === "point" ? location.point : null,
        label: `${speciesName(result.species, result.label)} ${percentText(result.score)}`, derived: true}] : [];
      overlay = result.saliency ? {...placement, saliency: result.saliency, marks} : null;
      drawOverlay();
      if (sampleId && samples.byId.has(sampleId) && (samples.model === null || samples.model === state.model)) {
        samples.model = state.model; samples.results.set(sampleId, result); renderSampleTiles();
      }
      state.completed += 1; $("requestCount").textContent = `${state.completed} completed`;
      $("runStatus").textContent = "Classified";
      if (autoSpeak && result.species) speakText(speciesName(result.species));
      return result;
    } catch (err) {
      if (state.abort !== controller) return null;
      const skipped = trigger === "live" ? skippedSampleReason(err) : null;
      if (err.name === "AbortError" || controller.signal.aborted) $("runStatus").textContent = "Stopped";
      else if (skipped) {
        if (counted) state.sampled -= 1;
        state.skipped += 1;
        $("runStatus").textContent = `Skipped · ${skipped}`;
        renderLiveCounts();
      } else {
        error(err.message); $("runStatus").textContent = "Request failed";
        if (!keepSource) state.running = false;
      }
      return null;
    } finally {
      engineRequests.release(ticket);
      if (state.abort === controller) {
        state.abort = null; state.busy = false; state.activeTrigger = null;
        if (!state.running) releaseCamera();
        controls();
      }
    }
  }
  // The latest saliency over the preview: the grid stretched over the picture the classifier was
  // sent (overlayPlacement), clipped to the visible picture, smoothed by the browser's scaling.
  // The latest overlay over the preview: a classifier's saliency heatmap, boxes and points, or
  // both. Everything is normalized over the picture the model was sent, so it is stretched over that
  // (overlayPlacement) and clipped to the visible picture. Marks a model wrote are solid; marks
  // derived from a classifier's saliency are dashed, because they are an estimate, not a detection.
  function drawOverlay() {
    const layer = $("overlayCanvas"), box = layer.parentElement;
    const show = Boolean(state.overlays && overlay && (overlay.saliency || overlay.marks?.length) &&
      overlay.natural.width && overlay.natural.height);
    layer.hidden = !show;
    const help = [];
    if (overlay?.saliency) {
      help.push(`${overlay.saliency.method ? `Saliency: ${overlay.saliency.method}. ` : ""}The heatmap shows where the classifier's evidence came from: brighter is more. It is not a detection.`);
    }
    if (overlay?.marks?.some(mark => mark.derived)) {
      help.push("The dashed box or point is derived from that heatmap - the region around its peak - so it is an estimate of where the Pokémon is, not a detection: none of these classifiers outputs boxes or points.");
    }
    if (overlay?.marks?.some(mark => !mark.derived)) {
      help.push("The box or point is Cosmos3-Edge's own 2D grounding: coordinates it wrote in its answer, normalized 0-1000 over the picture it was sent. Its percentage is how likely the model found the name it wrote.");
    }
    if (!help.length && state.kind === "classifier") {
      help.push("None of these classifiers outputs boxes, points or masks: their 2D grounding is a saliency heatmap, and the Name the Pokémon presets add a box or point derived from it.");
    }
    $("overlayHelp").hidden = !help.length;
    $("overlayHelp").textContent = help.join(" ");
    if (!show) return;
    const width = box.clientWidth, height = box.clientHeight, ratio = window.devicePixelRatio || 1;
    layer.width = Math.max(1, Math.round(width * ratio)); layer.height = Math.max(1, Math.round(height * ratio));
    const context = layer.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const {clip, grid} = overlayPlacement(overlay.sent, overlay.natural, width, height);
    context.save();
    context.beginPath(); context.rect(clip.x, clip.y, clip.width, clip.height); context.clip();
    if (overlay.saliency) {
      const {w, h, cells} = overlay.saliency;
      const tile = document.createElement("canvas"); tile.width = w; tile.height = h;
      const pixels = tile.getContext("2d").createImageData(w, h);
      const strength = overlay.marks?.length ? 140 : 200;
      cells.forEach((value, i) => pixels.data.set([...heatColor(value), Math.round(strength * value ** 1.3)], i * 4));
      tile.getContext("2d").putImageData(pixels, 0, 0);
      context.imageSmoothingEnabled = true; context.imageSmoothingQuality = "high";
      context.drawImage(tile, grid.x, grid.y, grid.width, grid.height);
    }
    const at = (x, y) => [grid.x + x * grid.width, grid.y + y * grid.height];
    context.font = "600 13px Inter, ui-sans-serif, sans-serif";
    for (const mark of overlay.marks || []) {
      const color = mark.derived ? "#7dd3fc" : "#b4f679";
      context.setLineDash(mark.derived ? [7, 5] : []);
      let labelAt = null;
      if (mark.box) {
        const [x1, y1] = at(mark.box[0], mark.box[1]), [x2, y2] = at(mark.box[2], mark.box[3]);
        for (const [lineWidth, stroke] of [[5, "rgba(0,0,0,0.6)"], [2.5, color]]) {
          context.lineWidth = lineWidth; context.strokeStyle = stroke; context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }
        labelAt = [x1, y1];
      }
      if (mark.point) {
        const [x, y] = at(mark.point[0], mark.point[1]);
        context.setLineDash([]);
        for (const [lineWidth, stroke] of [[5, "rgba(0,0,0,0.6)"], [2.5, color]]) {
          context.lineWidth = lineWidth; context.strokeStyle = stroke;
          context.beginPath(); context.arc(x, y, 9, 0, 2 * Math.PI); context.stroke();
          context.beginPath(); context.moveTo(x - 15, y); context.lineTo(x + 15, y); context.moveTo(x, y - 15); context.lineTo(x, y + 15); context.stroke();
        }
        labelAt ??= [x + 14, y - 14];
      }
      if (mark.label && labelAt) {
        const textWidth = context.measureText(mark.label).width;
        const lx = Math.min(Math.max(labelAt[0], clip.x), clip.x + clip.width - textWidth - 10);
        const ly = Math.max(labelAt[1] - 22, clip.y);
        context.setLineDash([]);
        context.fillStyle = "rgba(10,16,12,0.82)"; context.fillRect(lx, ly, textWidth + 10, 20);
        context.fillStyle = color; context.fillText(mark.label, lx + 5, ly + 15);
      }
    }
    context.restore();
  }
  new ResizeObserver(drawOverlay).observe($("overlayCanvas").parentElement);
  $("overlayToggleButton").addEventListener("click", () => {
    state.overlays = !state.overlays; drawOverlay(); controls();
  });

  // Sample results are one model's, and for Cosmos3-Edge one prompt's: a new prompt starts over.
  function resetSampleResults() {
    if (!samples.model || state.kind === "classifier" || samples.model === `${state.model}|${$("prompt").value.trim()}`) return;
    if (samples.run) samples.run.cancelled = true;
    samples.results.clear(); samples.model = null; renderSampleTiles();
  }
  // The sample set: thumbnails anyone can open in the preview, and - with a classifier selected -
  // a run over every image, scored against its label as each answer lands.
  // model: the selected classifier's id, for which species it can name; null for a Cosmos engine.
  // Only the latest request lands: a quick second switch must not get the first one's coverage.
  let samplesRequest = 0;
  async function loadSamples(model) {
    const request = ++samplesRequest;
    try {
      const response = await fetch(`/api/samples${model ? `?model=${encodeURIComponent(model)}` : ""}`, {cache: "no-store"});
      const data = await response.json();
      if (request !== samplesRequest) return;
      if (!response.ok || !Array.isArray(data?.images)) throw new Error("Sample set unavailable");
      samples.list = data.images.filter(item => typeof item?.id === "string" && typeof item.species === "string");
      samples.byId = new Map(samples.list.map(item => [item.id, item]));
      $("samplePanel").hidden = !data.configured;
      $("sampleGrid").replaceChildren(...samples.list.map(sampleTile));
      renderSampleTiles();
    } catch (_) { /* keep what was shown; the next model change asks again */ }
    controls();
  }
  function sampleTile(item) {
    const tile = document.createElement("button");
    tile.type = "button"; tile.className = "sample-tile"; tile.dataset.sampleId = item.id;
    tile.title = [item.credit, item.license].filter(Boolean).join(" · ");
    const image = document.createElement("img");
    image.loading = "lazy"; image.alt = ""; image.src = `/api/samples/${encodeURIComponent(item.id)}`;
    const truth = document.createElement("span"); truth.className = "truth"; truth.textContent = speciesName(item.species);
    const verdict = document.createElement("span"); verdict.className = "verdict";
    tile.append(image, truth, verdict);
    tile.addEventListener("click", () => showSample(item.id));
    return tile;
  }
  function renderSampleTiles() {
    const rows = [];
    for (const tile of $("sampleGrid").children) {
      const item = samples.byId.get(tile.dataset.sampleId), result = samples.results.get(tile.dataset.sampleId);
      if (!item) continue;
      const verdict = tile.querySelector(".verdict");
      const uncovered = state.kind === "classifier" && item.covered === false;
      tile.classList.toggle("selected", item.id === state.sampleId);
      tile.classList.toggle("uncovered", uncovered);
      tile.classList.toggle("correct", Boolean(result) && result.species === item.species);
      tile.classList.toggle("wrong", Boolean(result) && result.species !== item.species);
      if (result) {
        rows.push({truth: item.species, covered: item.covered, ranked: result.topk.map(r => r.species), score: result.score});
        verdict.textContent = `${result.species === item.species ? "✓" : "✗"} ${speciesName(result.species, result.label)}` +
          (typeof result.score === "number" ? ` ${percentText(result.score)}` : "");
      } else verdict.textContent = uncovered ? "Not a species it knows" : "";
      tile.hidden = $("sampleMistakes").checked && !(result && result.species !== item.species);
    }
    const score = scoreSamples(rows), pct = (a, b) => b ? ` (${(100 * a / b).toFixed(1)}%)` : "";
    const name = engineData?.active?.name || "the selected model";
    // How sure the model said it was, against whether it was right: a quick look at calibration.
    const mean = list => list.length ? percentText(list.reduce((a, b) => a + b, 0) / list.length) : null;
    const scored = rows.filter(row => typeof row.score === "number");
    const sureRight = mean(scored.filter(row => row.ranked[0] === row.truth).map(row => row.score));
    const sureWrong = mean(scored.filter(row => row.ranked[0] !== row.truth).map(row => row.score));
    $("sampleScore").hidden = !score.n;
    $("sampleScore").textContent = !score.n ? "" :
      `${name}: top-1 ${score.top1}/${score.n}${pct(score.top1, score.n)}` +
      (rows.some(row => row.ranked.length > 1) ? ` · top-5 ${score.top5}/${score.n}${pct(score.top5, score.n)}` : "") +
      (score.coveredN < score.n ? ` · on the ${score.coveredN} images of species it can name: top-1 ${score.coveredTop1}/${score.coveredN}${pct(score.coveredTop1, score.coveredN)}` : "") +
      (sureRight || sureWrong ? ` · its own probability averaged ${sureRight ?? "—"} when right, ${sureWrong ?? "—"} when wrong` : "");
    const covered = samples.list.filter(item => item.covered !== false).length;
    $("sampleStatus").textContent = samples.run ? `Running ${samples.results.size} of ${samples.list.length}…`
      : `${samples.list.length} images · ${new Set(samples.list.map(item => item.species)).size} species` +
        (state.kind === "classifier" && covered < samples.list.length ? ` · ${covered} of a species this classifier can name` : "");
  }
  async function showSample(id) {
    const item = samples.byId.get(id);
    if (!item || state.busy) return;
    reachyRequested = false;
    stop(); error();
    if (state.imageURL?.startsWith("blob:")) URL.revokeObjectURL(state.imageURL);
    state.imageURL = `/api/samples/${encodeURIComponent(id)}`; state.sampleId = id;
    overlay = null; drawOverlay(); renderSampleTiles();
    $("uploadedImage").src = state.imageURL;
    try { await $("uploadedImage").decode(); } catch (_) { error("This sample could not be decoded."); return; }
    if (state.sampleId !== id) return;
    $("uploadedImage").hidden = false; $("video").hidden = true; $("placeholder").hidden = true;
    $("sourceStatus").textContent = `Sample · ${speciesName(item.species)}`; controls();
    if (state.ready && !switchingEngine()) await analyze($("uploadedImage"));
  }
  // One answer read off the token stream without touching the caption: the text and its per-token
  // logprobs. Used by the sample run; analyze() streams into the caption itself.
  async function streamAnswer(body, signal) {
    const response = await fetch("/v1/chat/completions", {
      method: "POST", headers: {"Content-Type": "application/json"}, signal, body: JSON.stringify(body)});
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.error?.message || `Backend returned HTTP ${response.status}.`);
    }
    if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body) {
      throw new Error("Backend did not return a token stream.");
    }
    let output = "", done = false, finishReason = null;
    const logprobs = [];
    const parser = new SSEParser(event => {
      const completion = readCompletionEvent(event);
      if (completion.done) { done = true; return; }
      if (completion.finishReason) finishReason = completion.finishReason;
      if (completion.logprobs) logprobs.push(...completion.logprobs);
      if (typeof completion.text === "string") output += completion.text;
    });
    const reader = response.body.getReader(), decoder = new TextDecoder();
    try {
      while (!done) {
        const chunk = await reader.read();
        if (chunk.done) { parser.feed(decoder.decode(), true); break; }
        parser.feed(decoder.decode(chunk.value, {stream: true}));
      }
    } finally { reader.cancel().catch(() => {}); }
    if (!done || !["stop", "length"].includes(finishReason)) throw new Error("The answer did not complete.");
    return {output, logprobs};
  }
  // Every sample through the selected model, scored as each answer lands: a classifier by sample id
  // (the server's own copy of the file), or Cosmos3-Edge with a Name-the-Pokémon preset, sent the
  // way this page sends any picture, with that preset's prompt and the current settings.
  async function runSamples() {
    const prompt = $("prompt").value.trim(), pokemon = pokemonPreset(prompt);
    const classifier = state.kind === "classifier";
    if (!(classifier || pokemon) || !state.ready || state.busy || samples.run || switchingEngine()) return;
    let advanced;
    try { advanced = advancedValues(); } catch (err) { error(err.message); return; }
    stop(); error();
    const run = samples.run = {cancelled: false, model: state.model};
    const ticket = engineRequests.begin();
    samples.results.clear(); samples.model = classifier ? run.model : `${run.model}|${prompt}`;
    state.busy = true; renderSampleTiles(); controls();
    try {
      for (const item of samples.list) {
        if (run.cancelled || !engineRequests.current(ticket)) break;
        let result;
        if (classifier) {
          const response = await fetch("/api/classify", {
            method: "POST", headers: {"Content-Type": "application/json"}, signal: ticket.controller.signal,
            body: JSON.stringify({model: run.model, sample: item.id, saliency: false, topk: 5})
          });
          const body = await response.json().catch(() => null);
          if (!response.ok) throw new Error(body?.error?.message || `Classifier returned HTTP ${response.status}.`);
          result = readClassification(body);
        } else {
          const picture = new Image();
          picture.src = `/api/samples/${encodeURIComponent(item.id)}`;
          await picture.decode();
          const image = capture(picture);
          const answer = await streamAnswer({model: run.model, stream: true,
            temperature: 0, top_p: advanced.topP,
            max_image_tokens_per_image: advanced.imageTokens, stream_options: {include_usage: true},
            logprobs: true, top_logprobs: POKEMON_TOP_LOGPROBS, max_tokens: Number($("maxTokens").value),
            messages: [{role: "user", content: [{type: "text", text: prompt}, {type: "image_url", image_url: {url: image.url}}]}]},
            ticket.controller.signal);
          result = cosmosResult(readPokemonAnswer(answer.output, answer.logprobs));
        }
        if (run.cancelled || !engineRequests.current(ticket)) break;
        samples.results.set(item.id, result);
        renderSampleTiles();
      }
    } catch (err) {
      if (err.name !== "AbortError" && !run.cancelled) error(`Sample run stopped: ${err.message}`);
    } finally {
      engineRequests.release(ticket);
      if (samples.run === run) samples.run = null;
      state.busy = false; renderSampleTiles(); controls();
    }
  }
  $("sampleRunAll").addEventListener("click", runSamples);
  $("sampleStop").addEventListener("click", () => { if (samples.run) samples.run.cancelled = true; });
  $("sampleMistakes").addEventListener("change", renderSampleTiles);

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
    overlay = null; drawOverlay();
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
  function stop() {
    state.running = false; state.cameraGeneration += 1; state.abort?.abort();
    if (samples.run) samples.run.cancelled = true;
    releaseCamera(); controls();
  }
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
    overlay = null; drawOverlay();
    state.running = true; state.source = "camera"; const generation = ++state.cameraGeneration; controls();
    // Visible right next to the preview the user is looking at - not just the catch block's
    // error() below, which renders far away in the output panel and is easy to miss entirely
    // (reported bug: pressing Start appeared to do nothing but briefly re-layout the buttons).
    $("sourceStatus").textContent = "Starting camera…";
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
    } catch (err) {
      if (generation === state.cameraGeneration) {
        const message = `Camera unavailable: ${err.message}`;
        stop();  // stop() -> releaseCamera() would otherwise overwrite sourceStatus with a
                 // generic "Camera stopped", erasing the reason right after we show it.
        $("sourceStatus").textContent = message;
        error(`${message}. You can choose an image instead.`);
      }
    }
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
    if (state.imageURL?.startsWith("blob:")) URL.revokeObjectURL(state.imageURL);
    state.imageURL = URL.createObjectURL(file); state.sampleId = null;
    overlay = null; drawOverlay(); renderSampleTiles();
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
  // A discrete step per press, held to repeat - lands exactly on a multiple of stepDeg every
  // time (floor/ceil on the current value, not an accumulated +=), so repeated presses reach
  // named angles like 30/45/60/75/90 exactly regardless of whatever off-grid value a pad drag
  // left behind. sign follows the same invX/invY convention bindPad above was given: a jog
  // button moves the same way as dragging the pad in that direction would.
  function bindJog(id, key, stepDeg, sign, limitRad, padId) {
    const btn = $(id); if (!btn) return;
    let timer = null;
    const tick = () => {
      const stepRad = stepDeg * D2R;
      const n = sign > 0 ? Math.floor(T[key] / stepRad) + 1 : Math.ceil(T[key] / stepRad) - 1;
      T[key] = Math.max(-limitRad, Math.min(limitRad, n * stepRad));
      tTouch();
      const pad = $(padId); if (pad && pad._paint) pad._paint();
      tSend();
    };
    const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
    const start = ev => {
      ev.preventDefault();
      // Without this, a finger sliding off a small button while held can mean the browser never
      // delivers pointerup/pointerleave to it - the timer below then keeps firing indefinitely,
      // continuously sending a stale jog target that fights anything else asked of the robot
      // (Centre included) until the page is reloaded. bindPad's drag already does this for the
      // same reason; this had been missed here.
      try { btn.setPointerCapture(ev.pointerId); } catch (_) { /* unsupported is not fatal */ }
      tick();
      timer = setInterval(tick, 280);
      // Belt and suspenders: cap how long one press can run even if capture itself fails on some
      // browser/input combination, rather than leaving a stuck timer jogging the robot forever.
      setTimeout(stop, 4000);
    };
    btn.addEventListener("pointerdown", start);
    btn.addEventListener("pointerup", stop);
    btn.addEventListener("pointerleave", stop);
    btn.addEventListener("pointercancel", stop);
  }
  // Body yaw drives the head yaw target to match (so the two swing together as one turn, rather
  // than the base rotating while the head stays fixed relative to it - awkward to watch and to
  // demo). Negated: the slider's own left=-max/right=+max sense is unrelated to the pad's, and
  // matching them (slider right turns the same way dragging the pad right would) needs the sign
  // flipped, since positive slider values already share more in common with a leftward pad drag
  // (see bindPad's invX on padPY above) than a rightward one.
  //
  // Snaps at every 15deg (0/15/30/45/.../150 and their negatives, covering 30/45/60/75/90 and
  // more): inside a small zone around a snap point, a drag holds at the exact value instead of
  // tracking the pointer 1:1, the way a physical detent does, so a specific named angle is easy
  // to hit exactly rather than eyeballed against the readout.
  const BODY_YAW_SNAP_STEP_DEG = 15;
  const BODY_YAW_SNAP_TOLERANCE_DEG = 3;
  function snapBodyYawRad(rad) {
    const deg = rad / D2R;
    const nearest = Math.round(deg / BODY_YAW_SNAP_STEP_DEG) * BODY_YAW_SNAP_STEP_DEG;
    return Math.abs(deg - nearest) <= BODY_YAW_SNAP_TOLERANCE_DEG ? nearest * D2R : rad;
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
      T.yaw = Math.max(-Math.PI, Math.min(Math.PI, -snapped));
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
  // Both axes flipped from the original mapping - reported backwards on the physical
  // robot for set_target's target_head_pose (a different daemon route than the goto()-based
  // look()/centre() this sign convention was originally verified against - see Reachy.look()'s
  // own docstring - so the two need not agree).
  bindPad("padPY", "pyv", "yaw", "pitch", Math.PI, 0.7, true, false);
  bindJog("jogPitchUp", "pitch", 15, 1, 0.7, "padPY");
  bindJog("jogPitchDown", "pitch", 15, -1, 0.7, "padPY");
  bindJog("jogYawLeft", "yaw", 15, 1, Math.PI, "padPY");
  bindJog("jogYawRight", "yaw", 15, -1, Math.PI, "padPY");

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
      $("autoSpeakButton").hidden = $("autoSpeakHelp").hidden = !(st.enabled && st.speech_enabled);
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
  // Stiff/Soft/Limp, not a dropdown. Centre and the pose pad/sliders below silently refuse every
  // move while motors are Limp (see Reachy.set_target's own "motors are disabled" check in
  // reachy/reachy.py), so this needs to read as a mode to actively pick, not a setting to notice
  // was wrong afterward.
  const MOTOR_BUTTONS = [["reachyMotorStiff", "enabled"], ["reachyMotorSoft", "gravity_compensation"], ["reachyMotorLimp", "disabled"]];
  for (const [id, mode] of MOTOR_BUTTONS) $(id).addEventListener("click", () => reachyPost(`/api/reachy/motors/${mode}`));
  $("reachyAppStart").addEventListener("click", () => {
    const name = $("reachyAppSelect").value;
    if (name) reachyPost(`/api/reachy/apps/start/${encodeURIComponent(name)}`);
  });
  $("reachyAppStop").addEventListener("click", () => reachyPost("/api/reachy/apps/stop"));
  $("reachySpeakerVol").addEventListener("change", e => reachyPost(`/api/reachy/volume/speaker/${e.target.value}`));
  $("reachyMicVol").addEventListener("change", e => reachyPost(`/api/reachy/volume/mic/${e.target.value}`));
  // Piper.synth() (serve_ui.py) holds one lock for the whole synth+play call, so a second speak
  // request fired before the first finishes does not run alongside it - it queues behind it. A
  // click while one is already in flight used to do exactly that, and with live captioning
  // updating every second or two, a queued click could end up speaking a caption from several
  // answers ago by the time its turn came - "lag" that compounded the more it was clicked.
  // Disabling both buttons for the duration of a request keeps at most one in flight at a time,
  // so a request always speaks the caption it was asked to, promptly, or not at all.
  let speaking = false, speechRate = 1;
  // speaking also gates auto-speak (below): a caption that completes while Reachy is still
  // speaking the previous one is skipped rather than queued, the same reasoning as the button
  // guard - speech falls behind captioning by at most one utterance, never a growing backlog.
  async function speakText(text) {
    if (speaking || !text) return;
    speaking = true;
    const answerBtn = $("reachySpeakAnswer"), manualBtn = $("reachySpeakButton");
    const answerLabel = answerBtn.textContent, manualLabel = manualBtn.textContent;
    answerBtn.disabled = true; manualBtn.disabled = true;
    answerBtn.textContent = "Speaking…"; manualBtn.textContent = "Speaking…";
    try {
      await reachyPost("/api/reachy/speak", {text, rate: speechRate});
    } catch (_) { /* surfaced already via error() */ } finally {
      speaking = false;
      answerBtn.disabled = false; manualBtn.disabled = false;
      answerBtn.textContent = answerLabel; manualBtn.textContent = manualLabel;
    }
  }
  function setSpeechRate(rate) {
    speechRate = rate;
    for (const btn of document.querySelectorAll("#reachyControls .speedopt")) {
      btn.classList.toggle("active", parseFloat(btn.dataset.rate) === rate);
    }
  }
  for (const btn of document.querySelectorAll("#reachyControls .speedopt")) {
    btn.addEventListener("click", () => setSpeechRate(parseFloat(btn.dataset.rate)));
  }
  // Auto-speak speaks every caption at the selected speech rate and never changes it: the rate
  // active when it is switched on, or any rate picked while it runs, is the rate used.
  let autoSpeak = false;
  $("autoSpeakButton").addEventListener("click", () => {
    autoSpeak = !autoSpeak;
    $("autoSpeakButton").setAttribute("aria-pressed", String(autoSpeak));
    $("autoSpeakButton").textContent = `Auto-speak: ${autoSpeak ? "On" : "Off"}`;
  });
  $("reachySpeakButton").addEventListener("click", () => {
    const text = $("reachySpeakText").value.trim();
    if (text) speakText(text);
  });
  $("reachySpeakAnswer").addEventListener("click", () => {
    if (!state.completed || $("answer").classList.contains("streaming")) {
      error("No completed caption to speak yet - wait for one to finish.");
      return;
    }
    const text = $("answer").textContent.trim();
    if (text) speakText(text);
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
    const classifiers = engineData.engines?.some(engine => engine.kind === "classifier");
    $("engineSwitchHint").textContent = !engineData.configured ? "Model switching is not configured on this device."
      : "Only one vision-language engine is resident at a time. Switching pauses inference while the new model loads; a failed switch attempts to restore the previous model." +
        (classifiers ? " Classifiers load beside it in seconds and leave it running." : "");
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
