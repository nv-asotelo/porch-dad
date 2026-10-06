"use strict";
// Spot sets - where one target is in each picture: crops, overlaps, verdicts and scores. Synthetic;
// no device, backend or model.
const test = require("node:test");
const assert = require("node:assert/strict");
const {cropRect, boxIoU, toPicture, onTarget, spotResult, spotFromAnswer, spotVerdict, scoreSpots,
  readNameAnswer} = require("../web/app.js");

const close = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-9);

test("a detector box is cropped exactly as the Bernese mode's breed scorer crops it", () => {
  // breed_service.py: the box plus 10% of its longer side, clamped, truncated to whole pixels.
  // 960x540, box 96..192 x 54..162 (96x108 px): margin 10.8 -> crop (85, 43) to (202, 172).
  assert.deepEqual(cropRect([0.1, 0.1, 0.2, 0.3], 960, 540), [85, 43, 117, 129]);
  // Clamped at the picture's edges.
  assert.deepEqual(cropRect([0, 0, 0.5, 1], 100, 50), [0, 0, 55, 50]);
  assert.deepEqual(cropRect([0.9, 0.9, 1, 1], 100, 100), [89, 89, 11, 11]);
});

test("overlap, and a mark on the target by its box or its point", () => {
  assert.equal(boxIoU([0, 0, 1, 1], [0, 0, 1, 1]), 1);
  assert.equal(boxIoU([0, 0, 0.5, 1], [0.5, 0, 1, 1]), 0);
  assert.ok(Math.abs(boxIoU([0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]) - 1 / 7) < 1e-9);
  const target = [[0.4, 0.4, 0.6, 0.6]];
  assert.equal(onTarget({box: [0.41, 0.41, 0.6, 0.6]}, target), true);
  assert.equal(onTarget({box: [0.3, 0.3, 0.7, 0.7]}, target), false);       // IoU 0.25
  assert.equal(onTarget({point: [0.5, 0.45]}, target), true);
  assert.equal(onTarget({point: [0.1, 0.1]}, target), false);
  assert.equal(onTarget(null, target), false);
});

test("coordinates over the capture canvas become fractions of the picture", () => {
  const whole = {canvasWidth: 512, canvasHeight: 288, x: 0, y: 0, w: 512, h: 288};
  assert.ok(close(toPicture([0.1, 0.2, 0.3, 0.4], whole), [0.1, 0.2, 0.3, 0.4]));
  // A picture drawn at (16, 0, 480, 288) inside a 512x288 canvas (padded left and right).
  const padded = {canvasWidth: 512, canvasHeight: 288, x: 16, y: 0, w: 480, h: 288};
  assert.ok(close(toPicture([16 / 512, 0, 256 / 512, 1], padded), [0, 0, 0.5, 1]));
  assert.ok(close(toPicture([0, 0.5], padded), [0, 0.5]));                   // clamped into the picture
});

test("a classifier spots the target when a box it names the target is the target's own", () => {
  const dog = [[0.4, 0.1, 0.5, 0.2]];
  const boxes = [{box: [0.4, 0.1, 0.5, 0.2], species: "bernese-mountain-dog", label: "x", score: 0.8},
    {box: [0.1, 0.1, 0.2, 0.2], species: "bernese-mountain-dog", label: "x", score: 0.9},
    {box: [0.7, 0.7, 0.8, 0.8], species: "labrador-retriever", label: "x", score: 0.6}];
  assert.deepEqual(spotResult(dog, boxes, "bernese-mountain-dog"), {spot: true, named: true, right: true, score: 0.8, boxes});
  // Named, but only on another dog: not spotted.
  const wrong = spotResult(dog, [boxes[1], boxes[2]], "bernese-mountain-dog");
  assert.deepEqual([wrong.named, wrong.right, wrong.score], [true, false, 0.9]);
  // Not named at all, or nothing boxed.
  assert.deepEqual([spotResult(dog, [boxes[2]], "bernese-mountain-dog").named, spotResult(dog, [], "bernese-mountain-dog").score],
    [false, null]);
});

test("a VLM spots the target by naming it and boxing the right one", () => {
  const sent = {canvasWidth: 512, canvasHeight: 288, x: 0, y: 0, w: 512, h: 288};
  const dog = [[0.4, 0.1, 0.5, 0.2]];
  const answer = text => readNameAnswer(text, []);
  const on = spotFromAnswer(answer('{"name": "Bernese Mountain Dog", "bbox_2d": [400, 100, 500, 200]}'), sent, dog, "bernese-mountain-dog");
  assert.deepEqual([on.named, on.right, on.species], [true, true, "bernese-mountain-dog"]);
  const elsewhere = spotFromAnswer(answer('{"name": "Bernese mountain dog", "point_2d": [100, 900]}'), sent, dog, "bernese-mountain-dog");
  assert.deepEqual([elsewhere.named, elsewhere.right], [true, false]);
  const other = spotFromAnswer(answer('{"name": "Labrador Retriever", "bbox_2d": [400, 100, 500, 200]}'), sent, dog, "bernese-mountain-dog");
  assert.deepEqual([other.named, other.right], [false, false]);
  // Named, but no box or point to say where: not the wrong dog - it said nowhere.
  const nowhere = spotFromAnswer(answer('{"name": "Bernese Mountain Dog", "bbox_2d": null}'), sent, dog, "bernese-mountain-dog");
  assert.deepEqual([nowhere.named, nowhere.located, nowhere.right], [true, false, false]);
  assert.deepEqual(spotVerdict(true, nowhere), {ok: false, text: "✗ named it, no box or point"});
  assert.deepEqual([spotFromAnswer(null, sent, dog, "bernese-mountain-dog").named], [false]);
});

test("verdicts and the run's score count spotted, wrong one, missed and false alarms", () => {
  assert.deepEqual(spotVerdict(true, {named: true, right: true, score: 0.83}), {ok: true, text: "✓ spotted 83%"});
  assert.deepEqual(spotVerdict(true, {named: true, right: false}), {ok: false, text: "✗ named the wrong one"});
  assert.deepEqual(spotVerdict(true, {named: false, right: false}), {ok: false, text: "✗ missed"});
  assert.deepEqual(spotVerdict(false, {named: true, right: false}), {ok: false, text: "✗ false alarm"});
  assert.deepEqual(spotVerdict(false, {named: false, right: false}), {ok: true, text: "✓ not here"});
  assert.deepEqual(scoreSpots([
    {present: true, named: true, right: true, boxed: true}, {present: true, named: true, right: false, boxed: true},
    {present: true, named: true, located: false, right: false, boxed: true},
    {present: true, named: false, right: false, boxed: false}, {present: false, named: true, right: false},
    {present: false, named: false, right: false}]),
  {present: 4, spotted: 1, wrongOne: 1, unlocated: 1, missed: 1, boxed: 3, absent: 2, falseAlarms: 1});
});
