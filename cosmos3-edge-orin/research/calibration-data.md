# Optional AWQ image calibration corpus

**Prepared, not used.** The ignored local manifest `data/calibration/images.jsonl`
contains 32 distinct natural COCO val2017 JPEGs (4,984,558 bytes total) for the
single optional AWQ candidate. No device access, model execution, quantization,
quality evaluation, or performance measurement was performed for this work.

The selector orders images by COCO image ID, requires the official image metadata
to identify license 4, then requires the original Flickr photo's current oEmbed
response to identify the same CC BY 2.0 license and supply creator, title, and
photo URL. It stops at 32. Historical/current license mismatches and unavailable
attribution were excluded. The frozen manifest includes exact image URLs,
original Flickr links, author/title credits, license link, byte count, dimensions,
SHA-256, and retrieval timestamps. Full attribution and the selection audit are
under `data/calibration/`; the images are not redistributed in the repository or
presentation.

COCO's annotation license is separate from each photograph's license. The
selection uses the individual image rights metadata, with an additional current
Flickr check. [COCO terms](https://cocodataset.org/#termsofuse),
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

The official COCO image hostname returned a TLS certificate-name mismatch. The
same COCO S3 bucket was accessed through the certificate-valid AWS endpoint;
normal TLS validation remained enabled and both original and download URLs were
recorded. The caption metadata entry was obtained with exact ZIP byte ranges and
CRC32 verification. Its locally measured SHA-256 is
`afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`.
This is an integrity record, not a publisher-signed hash.
[Official download page](https://cocodataset.org/#download).

All 32 JPEGs fully decoded, matched official metadata dimensions, and passed the
existing `scripts/quantize_cosmos3_awq.py` `corpus_rows` loader with distinct,
matching SHA-256 values. They have no exact byte overlap with the existing
held-out fixture PNGs. The frozen manifest SHA-256 is
`d808b2897ba8148017ae62a208d47d466c408258b7ea8cc48b9b30dd31d26229`.
See [preparation evidence](../results/calibration-data-preparation.json).

Use this corpus only for calibration, with the existing real-FP16 success or
documented-memory-failure gate. Preserve the manifest, attribution, and metadata
when copying it into the separately authorized auxiliary GPU task directory. A
32-image natural-image sample does not establish representative video quality;
quality comparisons must use a separate fixed evaluation set.
