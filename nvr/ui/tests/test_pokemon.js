"use strict";
// Name presets (Pokémon and dog breeds): grounding JSON, name probability, saliency-derived
// locations and sample results. Synthetic; no device, backend or model.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {speciesKey, parseGrounding, nameProbability, readNameAnswer, saliencyLocation, SALIENCY_LOCATION_PARAMS,
  cosmosResult, namePreset, presetFits, presetForSubject, NAME_PRESETS, NAME_SUBJECTS, SAMPLE_SUBJECT,
  readCompletionEvent} = require("../web/app.js");

const encoder = new TextEncoder();
const token = (text, probability) => ({token: text, logprob: Math.log(probability), bytes: [...encoder.encode(text)]});

test("names become the sample set's species identifiers", () => {
  for (const [name, key] of [["Pikachu", "pikachu"], ["Mr. Mime", "mr-mime"], ["Nidoran♀", "nidoran-f"],
    ["Farfetch'd", "farfetchd"], ["Flabébé", "flabebe"], ["Type: Null", "type-null"], [" Ho-Oh ", "ho-oh"],
    ["Mime Jr.", "mime-jr"], ["", null], [null, null]]) {
    assert.equal(speciesKey(name), key, String(name));
  }
});

test("grounding is read from every shape Cosmos3-Edge writes, as 0-1 fractions", () => {
  // Shapes seen in 3x189 real answers to the presets.
  const one = (name, box, point) => [{name, box, point}];
  assert.deepEqual(parseGrounding('{"name": "Pikachu", "bbox_2d": [263, 303, 563, 770]}'), one("Pikachu", [0.263, 0.303, 0.563, 0.77], null));
  assert.deepEqual(parseGrounding('{\n  "name": "Eevee",\n  "point_2d": [500, 600]\n}'), one("Eevee", null, [0.5, 0.6]));
  assert.deepEqual(parseGrounding("```json\n[\n  {\"bbox_2d\": [263, 303, 563, 770], \"label\": \"Pikachu\"}\n]\n```"),
    one("Pikachu", [0.263, 0.303, 0.563, 0.77], null));
  // A duplicated key: the model's first value is the one it meant (JSON.parse would keep the last).
  assert.deepEqual(parseGrounding('{"name": "Lucario", "bbox_2d": [300, 100, 700, 800], "bbox_2d": [250, 780, 750, 990]}')[0].box, [0.3, 0.1, 0.7, 0.8]);
  // Malformed or trailing JSON still yields the first answer.
  assert.deepEqual(parseGrounding('{"name": "Diglett", "bbox_2d": [[120, 340, 560, 900]]}')[0].box, [0.12, 0.34, 0.56, 0.9]);
  assert.deepEqual(parseGrounding('{"name": "Pinsir", "bbox_2d": [10, 20, 30, 40]}, {"name": "Pinsir", "bbox_2d": [5, 6')[0].box, [0.01, 0.02, 0.03, 0.04]);
  // Coordinates inside a nested object, and a box the model declined to give.
  assert.deepEqual(parseGrounding('{"name": "Gengar", "point_2d": [{"bbox_2d": [412, 555]}]}')[0].point, [0.412, 0.555]);
  assert.deepEqual(parseGrounding('{"name": "Oddish", "bbox_2d": null}'), one("Oddish", null, null));
  // Names are cleaned, coordinates clamped, inverted boxes dropped.
  assert.equal(parseGrounding('{"name": " Pikachu. ", "point_2d": [1, 2]}')[0].name, "Pikachu");
  assert.deepEqual(parseGrounding('{"name": "Psyduck", "bbox_2d": [-20, 0, 1100, 900]}')[0].box, [0, 0, 1, 0.9]);
  assert.equal(parseGrounding('{"name": "X", "bbox_2d": [600, 0, 500, 900]}')[0].box, null);
  for (const text of ["A Pikachu figure.", "[1, 2, 3, 4]", "{not json}", "", null]) {
    assert.equal(parseGrounding(text), null, String(text));
  }
});

test("the name's probability multiplies its own tokens, and errs low at a straddling edge", () => {
  const exact = [token('{"name": "', 0.99), token("Pik", 0.8), token("achu", 0.9), token('", "bbox_2d": [1, 2, 3, 4]}', 0.95)];
  assert.ok(Math.abs(nameProbability(exact, "Pikachu") - 0.72) < 1e-9);
  const straddling = [token('{"name": "Pi', 0.5), token("kachu", 0.9), token('"}', 0.99)];
  assert.ok(Math.abs(nameProbability(straddling, "Pikachu") - 0.45) < 1e-9);
  // A multi-byte character split across two tokens still lines up.
  const flabebe = encoder.encode("Flabébé");
  const split = [{token: "", logprob: Math.log(0.5), bytes: [...flabebe.slice(0, 5)]}, {token: "", logprob: Math.log(0.5), bytes: [...flabebe.slice(5)]}];
  assert.ok(Math.abs(nameProbability(split, "Flabébé") - 0.25) < 1e-9);
  assert.equal(nameProbability(exact, "Raichu"), null);
  // A sampled token outside the returned top-K has no bytes: no probability rather than a wrong one.
  assert.equal(nameProbability([...exact, {token: "?", logprob: null, bytes: null}], "Pikachu"), null);
  assert.equal(nameProbability([], "Pikachu"), null);
});

test("an answer becomes a caption, labelled marks, and a sample result", () => {
  const text = '{"name": "Pikachu", "bbox_2d": [263, 303, 563, 770]}';
  const read = readNameAnswer(text, [token('{"name": "', 0.99), token("Pikachu", 0.87), token('", "bbox_2d": [263, 303, 563, 770]}', 0.9)]);
  assert.equal(read.caption, "Pikachu · 87%");
  assert.deepEqual(read.marks, [{box: [0.263, 0.303, 0.563, 0.77], point: null, label: "Pikachu 87%", derived: false}]);
  assert.deepEqual(cosmosResult(read), {species: "pikachu", label: "Pikachu", score: 0.87,
    topk: [{species: "pikachu", label: "Pikachu", score: 0.87}], cosmos: true});
  const withoutLogprobs = readNameAnswer(text, []);
  assert.equal(withoutLogprobs.caption, "Pikachu");
  assert.equal(cosmosResult(withoutLogprobs).score, null);
  assert.equal(cosmosResult(null).species, null);
  assert.equal(readNameAnswer("It is a Pikachu.", []), null);
});

test("a saliency grid becomes a box and a point, as the Python reference derives them", () => {
  const plain = {threshold: 0.5, relative: "max", upsample: 1, connectivity: 4, component: "peak", pad: 0, point: "peak"};
  const grid = (w, h, hot) => ({w, h, cells: Array.from({length: w * h}, (_, i) => hot[i] || 0)});
  assert.deepEqual(saliencyLocation(grid(7, 7, {24: 1}), plain), {box: [3 / 7, 3 / 7, 4 / 7, 4 / 7], point: [3.5 / 7, 3.5 / 7]});
  assert.deepEqual(saliencyLocation(grid(7, 7, {0: 1}), {...plain, pad: 0.5}).box, [0, 0, 1.5 / 7, 1.5 / 7]);
  assert.equal(saliencyLocation(grid(3, 3, {}), plain), null);
  assert.equal(saliencyLocation({w: 2, h: 2, cells: [0.5, 0.5, 0.5, 0.5]}, plain), null);
  const strip = {w: 6, h: 1, cells: [1, 0, 0, 0.6, 0.6, 0.6]};
  assert.deepEqual(saliencyLocation(strip, plain).box, [0, 0, 1 / 6, 1]);
  assert.deepEqual(saliencyLocation(strip, {...plain, component: "largest"}).box, [3 / 6, 0, 1, 1]);
  assert.deepEqual(saliencyLocation(strip, {...plain, component: "all"}).box, [0, 0, 1, 1]);
  assert.throws(() => saliencyLocation(grid(2, 2, {0: 1}), {...plain, upsample: 2}), TypeError);
  // Real grids - the Pokémon sample set, and Commons photos for the dog-breed classifiers - against
  // the reference's exact output.
  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "saliency_location_cases.json"), "utf8"));
  assert.deepEqual(JSON.parse(JSON.stringify(SALIENCY_LOCATION_PARAMS)), fixture.params);
  // Dog-Breed-120 derives no mark at all: its map is nearly flat.
  assert.equal(SALIENCY_LOCATION_PARAMS.dogbreed120, null);
  for (const c of fixture.cases) {
    const got = saliencyLocation(c.saliency, SALIENCY_LOCATION_PARAMS[c.model]);
    assert.deepEqual(got, c.expected, `${c.model} ${c.file}`);
    assert.ok(got.box[0] <= got.point[0] && got.point[0] <= got.box[2] && got.box[1] <= got.point[1] && got.point[1] <= got.box[3]);
  }
});

test("presets are recognised by their exact prompt and locate a box or a point, per subject", () => {
  assert.deepEqual(NAME_PRESETS.map(preset => `${preset.subject} ${preset.locate}`),
    ["pokemon box", "pokemon point", "dog box", "dog point"]);
  for (const preset of NAME_PRESETS) {
    assert.equal(namePreset(preset.prompt), preset);
    assert.ok(NAME_SUBJECTS[preset.subject], preset.label);
  }
  assert.equal(namePreset("Describe what you see in this image in one sentence."), null);
  assert.ok(NAME_SUBJECTS[SAMPLE_SUBJECT]);
});

test("a classifier answers its own subject's presets, and selecting it swaps in the same mark", () => {
  const [pokemonBox, pokemonPoint, dogBox, dogPoint] = NAME_PRESETS;
  assert.equal(presetFits(dogBox, "dog"), true);
  assert.equal(presetFits(pokemonBox, "dog"), false);
  // A registry entry without a subject takes any Name preset, as before subjects existed.
  assert.equal(presetFits(pokemonPoint, null), true);
  assert.equal(presetFits(null, "dog"), false);
  assert.equal(presetForSubject(pokemonBox.prompt, "dog"), dogBox);
  assert.equal(presetForSubject(dogPoint.prompt, "pokemon"), pokemonPoint);
  // Nothing to change: the preset already fits, the prompt is not a Name preset, or no subject.
  assert.equal(presetForSubject(dogBox.prompt, "dog"), null);
  assert.equal(presetForSubject("Describe the scene.", "dog"), null);
  assert.equal(presetForSubject(pokemonBox.prompt, null), null);
  assert.equal(presetForSubject(pokemonBox.prompt, "cat"), null);
  // Cosmos3-Edge reads a breed answer the same way it reads a Pokémon one.
  const read = readNameAnswer('{"name": "Bernese Mountain Dog", "bbox_2d": [120, 80, 640, 900]}', []);
  assert.deepEqual(read.marks[0].box, [0.12, 0.08, 0.64, 0.9]);
  assert.equal(cosmosResult(read).species, "bernese-mountain-dog");
});

test("streamed chunks carry their logprobs through", () => {
  const event = {data: JSON.stringify({choices: [{delta: {content: "Pik"}, logprobs: {content: [token("Pik", 0.8)]}}]})};
  assert.equal(readCompletionEvent(event).logprobs[0].token, "Pik");
  assert.equal(readCompletionEvent({data: JSON.stringify({choices: [{delta: {content: "x"}}]})}).logprobs, undefined);
});
