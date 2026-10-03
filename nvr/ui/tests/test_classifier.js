"use strict";
// Synthetic classifier-answer, overlay-placement and sample-score tests; no device, backend or model.
const test = require("node:test");
const assert = require("node:assert/strict");
const {speciesName, readClassification, describeClassification, overlayPlacement, heatColor,
  scoreSamples} = require("../web/app.js");

const answer = () => ({model: "vit", species: "mr-mime", label: "Mr. Mime", score: 0.91,
  topk: [{species: "mr-mime", label: "Mr. Mime", score: 0.91}, {species: null, label: "MissingNo", score: 0.04},
    {species: "jynx", label: "Jynx", score: 0.012}],
  saliency: {w: 2, h: 2, cells: [0, 0.25, 0.5, 1], method: "attention rollout"}, boxes: [],
  timing_ms: {preprocess: 3.1, inference: 18.4, total: 22.0}});

test("species names read as words, and an unmapped label keeps its own name", () => {
  assert.equal(speciesName("mr-mime"), "Mr Mime");
  assert.equal(speciesName("pikachu"), "Pikachu");
  assert.equal(speciesName(null, "MissingNo"), "MissingNo");
  assert.equal(speciesName(null), "Unknown");
});

test("a classifier answer is read strictly", () => {
  const result = readClassification(answer());
  assert.equal(result.species, "mr-mime");
  assert.equal(result.inferenceMs, 18.4);
  assert.deepEqual(result.saliency.cells, [0, 0.25, 0.5, 1]);
  for (const broken of [
    {...answer(), topk: []},
    {...answer(), topk: [{species: "x", score: 1.5}]},
    {...answer(), topk: [{species: "x", score: NaN}]},
    {...answer(), saliency: {w: 2, h: 2, cells: [0, 1, 1]}},
    {...answer(), saliency: {w: 2, h: 2, cells: [0, 1, 1, 2]}},
    {...answer(), saliency: {w: 0, h: 0, cells: []}},
  ]) assert.throws(() => readClassification(broken));
  assert.equal(readClassification({...answer(), saliency: null}).saliency, null);
});

test("the caption names the top species and the runners-up", () => {
  assert.equal(describeClassification(readClassification(answer())),
    "Mr Mime · 91%\n2. MissingNo 4.0% · 3. Jynx 1.2%");
});

test("saliency covers the picture the classifier saw, not the letterbox", () => {
  // A 640×480 picture shown in an 800×800 box: scaled 1.25, centred vertically.
  const whole = {canvasWidth: 640, canvasHeight: 480, x: 0, y: 0, w: 640, h: 480};
  const {clip, grid} = overlayPlacement(whole, {width: 640, height: 480}, 800, 800);
  assert.deepEqual(clip, {width: 800, height: 600, x: 0, y: 100});
  assert.deepEqual(grid, {x: 0, y: 100, width: 800, height: 600});
  // A 200×100 picture padded into a 256×256 capture canvas at (28, 78): the grid spans the canvas,
  // so it reaches past the visible picture by the padding, scaled the same way.
  const padded = {canvasWidth: 256, canvasHeight: 256, x: 28, y: 78, w: 200, h: 100};
  const placed = overlayPlacement(padded, {width: 200, height: 100}, 400, 200);
  assert.deepEqual(placed.clip, {width: 400, height: 200, x: 0, y: 0});
  assert.deepEqual(placed.grid, {x: -56, y: -156, width: 512, height: 512});
});

test("the heat ramp only gets brighter", () => {
  let previous = -1;
  for (let v = 0; v <= 1.0001; v += 0.05) {
    const [r, g, b] = heatColor(v), luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    assert.ok(luminance > previous, `luminance falls at ${v}`);
    previous = luminance;
  }
  assert.deepEqual(heatColor(-1), heatColor(0));
  assert.deepEqual(heatColor(2), heatColor(1));
});

test("a sample run counts top-1, top-5 and the covered subset", () => {
  const score = scoreSamples([
    {truth: "pikachu", covered: true, ranked: ["pikachu", "raichu"]},
    {truth: "eevee", covered: true, ranked: ["vaporeon", "jolteon", "flareon", "espeon", "eevee"]},
    {truth: "absol", covered: false, ranked: ["pikachu"]},
    {truth: "ditto", ranked: [null, "ditto"]},
  ]);
  assert.deepEqual(score, {n: 4, top1: 1, top5: 3, coveredN: 3, coveredTop1: 1});
});
