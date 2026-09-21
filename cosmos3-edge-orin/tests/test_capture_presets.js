"use strict";

// Pure browser-policy tests. No camera, model, GPU or hardware is exercised.
// FrameCadence decides eligibility only; observing while busy and dropping
// eligible frames belongs to the application's separate request lifecycle.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const {CAPTURE_PRESETS, FrameCadence} = require("../web/app.js");

assert.ok(CAPTURE_PRESETS, "web/app.js must export CAPTURE_PRESETS");
assert.equal(typeof FrameCadence, "function", "web/app.js must export FrameCadence");

function cameraVideo(preset) {
  assert.ok(preset.cameraConstraints, "Preset must include camera constraints");
  return preset.cameraConstraints.video ?? preset.cameraConstraints;
}

const live = CAPTURE_PRESETS["live-vlm"];
const lightweight = CAPTURE_PRESETS.lightweight;
assert.ok(live, "Upstream-style live preset exists");
assert.ok(lightweight, "Lightweight preset remains available");
assert.equal(live.temperature, 0.7);
assert.equal(live.jpegQuality, 0.75);
assert.equal(live.maxSide, null, "Live preset must not impose a client resize limit");
assert.deepEqual(cameraVideo(live).width, {ideal: 1280});
assert.deepEqual(cameraVideo(live).height, {ideal: 720});
assert.equal(Object.hasOwn(cameraVideo(live), "frameRate"), false,
  "Live preset must not retain the lightweight frame-rate constraint");
assert.equal(lightweight.temperature, 0);
assert.equal(lightweight.jpegQuality, 0.8);
assert.equal(lightweight.maxSide, 512);
assert.deepEqual(cameraVideo(lightweight).width, {ideal: 640});
assert.deepEqual(cameraVideo(lightweight).height, {ideal: 480});
assert.deepEqual(cameraVideo(lightweight).frameRate, {ideal: 15, max: 30});
const html = fs.readFileSync(require.resolve("../web/index.html"), "utf8");
assert.match(html, /id="lightweightPreset"[^>]*checked/, "First page load selects Lightweight");
assert.doesNotMatch(html, /id="liveVlmPreset"[^>]*checked/, "Live VLM remains an optional capture preset");
assert.match(html, /id="maxTokens"[^>]*value="64"/, "First demo answer limit is 64 output tokens");
assert.match(html, /<textarea id="prompt"[^>]*>Describe the visible scene in one concise sentence\. Focus on objects and actions\.<\/textarea>/);
assert.match(html, /<option value="1000" selected>1 second<\/option>/);
assert.match(html, /<option value="512" selected>512 px<\/option>/);
assert.doesNotMatch(html, /<select id="(?:interval|resolution)" disabled>/);
console.log("PASS: live and lightweight capture, sampling and JPEG preset contracts.");

// Eligibility follows presented-frame metadata, independent of wall-clock FPS.
// Simulate three seconds at several frame rates without creating real timers.
for (const fps of [15, 24, 30, 60]) {
  const cadence = new FrameCadence();
  const due = [];
  for (let frame = 1; frame <= fps * 3; frame += 1) {
    const eligible = cadence.observe(frame);
    assert.equal(typeof eligible, "boolean");
    if (eligible) due.push(frame);
  }
  assert.deepEqual(due, Array.from({length: Math.floor(fps * 3 / 30)}, (_, index) => (index + 1) * 30),
    `${fps} FPS still produces one eligible frame per 30 presented frames`);
}

function checkSequence(cadence, observations, label) {
  for (const [frame, expected] of observations) {
    assert.equal(cadence.observe(frame), expected, `${label}: observe(${String(frame)})`);
  }
}

// A callback may attach after the browser has already presented many frames.
// The first observation counts as frame one, rather than inheriting that past.
checkSequence(new FrameCadence(), [
  [100, false], [128, false], [129, true], [130, false], [158, false], [159, true]
], "Relative initial frame baseline");

checkSequence(new FrameCadence(3), [
  [8, false], [9, false], [10, true], [11, false], [13, true]
], "Explicit cadence interval");
checkSequence(new FrameCadence(1), [[8, true], [8, false], [9, true]],
  "Every-frame cadence includes the first valid observation and rejects duplicates");

// Several eligibility boundaries can elapse between callbacks. Coalesce them
// into one fresh decision, then wait for the next actual boundary: no backlog.
checkSequence(new FrameCadence(), [
  [1, false], [100, true], [100, false], [101, false], [119, false], [120, true],
  [1000, true], [1000, false], [1001, false], [1019, false], [1020, true]
], "Skipped callbacks coalesce instead of replaying old eligible frames");

const invalid = [undefined, null, NaN, Infinity, -Infinity, -1, 1.5, "30", false, {}, []];
const validAfterInvalid = new FrameCadence();
for (const frame of invalid) assert.equal(validAfterInvalid.observe(frame), false,
  `Invalid initial observation ${String(frame)} must not create a baseline`);
checkSequence(validAfterInvalid, [[20, false], [48, false]], "First valid baseline after invalid input");
for (const frame of invalid) assert.equal(validAfterInvalid.observe(frame), false,
  `Invalid observation ${String(frame)} must not advance or reset the counter`);
checkSequence(validAfterInvalid, [[48, false], [49, true], [49, false], [78, false], [79, true]],
  "Invalid and repeated observations preserve progress");

// A backwards valid counter means a new stream. It starts a fresh relative
// interval, even after the previous stream had reached an eligible boundary.
checkSequence(new FrameCadence(), [
  [500, false], [529, true], [530, false], [7, false], [7, false], [35, false],
  [36, true], [37, false], [3, false], [31, false], [32, true]
], "A new stream resets the frame baseline");

console.log("PASS: frame-based cadence across FPS, relative start, skipped callbacks, duplicate/invalid counts and stream resets.");
