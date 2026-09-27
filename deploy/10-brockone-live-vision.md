# Switch Live Vision between Cosmos3-Edge and brockone

Live Vision can proxy separate local model services while keeping only one
model resident. The model buttons show actual installation availability.
Switching pauses automatic capture, cancels the browser's old request and
discards late responses. The server serializes a switch with inference, stops
other registered models, starts the requested model and verifies its health
and model ID. If startup fails, it stops the failed model and restores the
previous service. The camera preview and Reachy services remain running.

This includes the four UI files from the deployed
`NV-livestream-reachy-cosmos-demo` commit `7a5dd9251bb12f090c559186321c3e7fbbf197b0`
and its Reachy client, without the demo branch's unrelated removals. Existing
camera, motor, microphone, TTS and service resource controls remain available.

## Local configuration

Copy [`engines.example.json`](../nvr/ui/config/engines.example.json) to a local
configuration file and set the real paths, unit names and unused loopback
ports. The example intentionally leaves brockone disabled. This repository
does not contain model weights, target engines, private qualification evidence
or credentials. Do not change the existing Cosmos engine symlink to brockone:
their plugin and runtime builds can differ.

Run the UI with its existing HTTPS/camera/Reachy/TTS arguments, plus:

```sh
python3 nvr/ui/scripts/serve_ui.py \
  --engines-config /path/to/local-engines.json --default-engine cosmos \
  --host 0.0.0.0 --port 8091 --https-port 8443 \
  --cert /path/to/cert.pem --key /path/to/key.pem --allow-insecure-lan
```

Keep each backend on loopback. The browser calls only the UI's HTTPS origin,
so switching needs no browser CORS exception or mixed HTTP/HTTPS requests.
The registry contains port numbers, never arbitrary upstream URLs. Engine
switch requests also require the same-origin token from `/api/access` in
`X-Reachy-Token`, in addition to the existing Host and origin checks.

The UI's Unix account needs passwordless `systemctl start` and `systemctl stop`
for **only the two configured model units**. For the example units/account,
validate an administrator-managed sudoers fragment with `visudo -cf`:

```sudoers
jetson ALL=(root) NOPASSWD: /usr/bin/systemctl start porch-dad-shim-v3.service, /usr/bin/systemctl stop porch-dad-shim-v3.service, /usr/bin/systemctl start brockone-edge.service, /usr/bin/systemctl stop brockone-edge.service
```

No generic service command, shell, symlink replacement or robot service
permission is required. Starting the UI only discovers the resident model;
it does not start or stop model services. Retain the existing Cosmos startup
policy and do not enable both model services at boot.

## Enable brockone after target qualification

Build the engine on the actual Orin, under its SD-backed
`/home/jetson/brockone` directory, with its separate qualified runtime and
serving environment. Preserve the existing Cosmos deployment. The brockone
endpoint must expose `/health`, `/v1/models` with ID `brockone`, and the
non-streaming `/v1/chat/completions` interface.

After actual engine loading, finite prediction and prepared-image checks
pass, record their evidence in the private target validation receipt. The
UI requires these fields and the exact receipt SHA256 in its local registry:

```json
{
  "state": "orin_engine_validated",
  "model_id": "brockone",
  "engine_root": "/home/jetson/brockone/engines/rtn168-head-fp16",
  "engine_load_passed": true,
  "finite_predictions_passed": true,
  "prepared_pixels_passed": true
}
```

These fields must describe real, referenced results. A successful compile or
training run is insufficient. The serving launcher separately verifies the
full engine/runtime/model provenance; this small UI receipt does not replace
those checks. Set `enabled` to `true`, update `readiness_sha256`, and restart
only the UI to read the configuration. Missing/changed proof disables
brockone even if an unqualified service happens to be listening.

brockone uses its fixed identification instruction, greedy decoding, a
64-token output cap and 512 image tokens. The UI shows and locks those
settings while it is selected, then restores the user's Cosmos settings on
switch-back. Its complete JSON answer is framed as one final SSE message for
the UI. This is **not token streaming**: no TTFT or native token rate is
fabricated; the UI labels its browser full-answer round trip separately.

## Verify and recover

```sh
python3 -m unittest discover -s nvr/ui/tests
node --test nvr/ui/tests/test_engine_switch.js
```

The tests use synthetic services and responses, not real model predictions.
On the device, check the HTTPS page, choose brockone, verify its actual model
ID and a permitted diagnostic image, then switch back to Cosmos3-Edge.
Confirm that the previous model unit stops, the selected one is healthy,
Reachy still works and timing averages reset. Preserve the old UI release,
unit configuration and engine registry for rollback. Actual device switch
and prediction evidence belongs in the private deployment report.
