# Natural-image smoke photographs

These three exact COCO JPEGs are provided for a final-selected-model functional illustration. They are disjoint by COCO ID and image SHA-256 from the 32 calibration images, and by SHA-256 from the six synthetic quality JPEGs and source PNGs. `manifest.json` contains the frozen prompts and nine observation rubrics, written after directly inspecting each image and before any inference. No model was run to create these fixtures.

This set is **not an optimization acceptance gate**. Run it only on the final selected model. Keep raw answers and distinguish observations from expected facts. Native dimensions differ, so any timing is a smoke sample, not a paired 512px benchmark.

## Image credits

- **01-market-produce**: [6.24.09 Market](https://www.flickr.com/photos/whitneyinchicago/3669674438) by [whitneyinchicago](https://www.flickr.com/photos/whitneyinchicago/), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). COCO val2017 ID 23781; 640 × 427 px. Exact COCO JPEG, no edits. SHA-256: `085b1c7f25bf1d149b6dff82dfdb0a83fbf6e875c01d5d374070eb582aada80e`. Saved Flickr evidence: [metadata/flickr-3669674438.json](metadata/flickr-3669674438.json).

- **02-cap-and-mitt**: [Beit Shemesh Hat and Mitt](https://www.flickr.com/photos/roncantrell/837387952) by [macman715](https://www.flickr.com/photos/roncantrell/), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). COCO val2017 ID 27932; 288 × 307 px. Exact COCO JPEG, no edits. SHA-256: `9ab6d8c5a124427d77add9827dc6ad58ce94efb003c34e5e20699acd82f43731`. Saved Flickr evidence: [metadata/flickr-837387952.json](metadata/flickr-837387952.json).

- **03-dog-on-bench**: [Rusty on the bench](https://www.flickr.com/photos/dharrels/4228514131) by [Dan Harrelson](https://www.flickr.com/photos/dharrels/), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). COCO val2017 ID 29393; 500 × 375 px. Exact COCO JPEG, no edits. SHA-256: `378763dec71a25c7cb44ff9e345c84ca0fe2d8bf993b88775c360549edf3b733`. Saved Flickr evidence: [metadata/flickr-4228514131.json](metadata/flickr-4228514131.json).


## Provenance and review

Public COCO per-image metadata supplies historical license ID 4; fresh Flickr oEmbed responses independently identify CC BY 2.0, the photographer, title and canonical photo page. These are saved in `metadata/`, including download records and the selected original COCO image entries. The original caption annotation SHA-256 is recorded; captions were used only to select subject matter. COCO’s annotation license does not supply the photographs’ image licenses.

CC BY 2.0 permits redistribution and adaptation with attribution; retain these image credits, source links, license links and any modification notices when reusing the photographs. The sports image’s existing photographer watermark is preserved. Neither the creators nor COCO endorse this experiment. [Creative Commons license deed](https://creativecommons.org/licenses/by/2.0/).

Visual review: market produce (broccoli, carrots, strawberries), cap above baseball mitt, and a standing dog on a bench. The photographs contain no visible person to identify. No breed, pet name, brand, text recognition or location is required. All three JPEGs were fully decoded with Pillow; dimensions, distinct hashes and disjointness were verified.
