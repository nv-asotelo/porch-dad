# Prepare and transfer the verified deployment

`scripts/deploy_transfer.py prepare` performs only local work. It rechecks every selected reasoner file against `results/model-download.json`, requires the pinned TensorRT-Edge-LLM checkout and its three exact clean submodule commits, and creates `data/deployment/manifest.json` plus a backend archive. No device, GPU host, network fetch, package installation or inference is involved.

```sh
python3 scripts/deploy_transfer.py prepare
```

The project files are explicitly allowlisted in the helper: original UI, benchmark, launcher, useful deployment documentation, tests, fixtures and attribution. The complete verified `models/cosmos3-edge-reasoner` selection is included with its original model notices. `.qa`, credentials, unrelated external projects, downloads, original Git configuration/hooks, virtual environments and build/engine caches are excluded. The backend is recreated from local public checkouts with sanitized Git metadata; its actual HEAD and initialized submodule revisions remain available to the launcher and build tools. Source URLs remain the corresponding public repositories. Preparation does not duplicate the 7.7 GB model; sending streams those verified files directly from their source paths. The backend archive is retained as a small separate preparation artifact.

The fresh backend clone also receives exactly the three saved source patches: Cosmos3 patch-projection layout, Cosmos-only half-pixel position coordinates, and the independent encoder-cache budget API. Preparation requires the reconstructed complete tracked diff to equal the current local backend; undocumented edits, missing patches and changed submodules fail instead of being omitted. The manifest records each patch hash, each changed source file's before/after hashes, and the complete binary Git diff hash. The target verifies patched file hashes and the same complete diff after extraction. Source patches remain visible as tracked edits above the original public commit; they are not presented as upstream commits.

The RTN converter/helper and its tests are included as original source, but the separately generated `models/cosmos3-edge-rtn-int4` candidate is **not** added to this deployment's original model-transfer scope. Its transfer and validation remain explicit subsequent operations.

The allowlist also includes the selected-profile service launcher, installer, runtime preflight/repair helpers and the official-reference quality diagnostic. `results/model-download.json` is the only included results receipt; it provides the verified source hashes needed for reproducible on-device RTN conversion. The three natural-image smoke JPEGs are included with their manifest, README credits and all saved CC BY 2.0 attribution metadata. They remain a final-model illustration, separate from optimization acceptance.

`deployment/selected.env` and `deployment/selected-config.json` are explicitly allowlisted, reviewed non-secret snapshots of this selected deployment. A fresh target must still build and validate its own matching engine profile; copied provenance alone does not establish local readiness. It must define `COSMOS_PROFILE` (`fp16` or `rtn-v1`), an absolute `COSMOS_MODEL_DIR`, and an absolute `COSMOS_CACHE_DIR` using systemd environment-file syntax (`NAME=value`, without `export`). For both profiles, preserve the same `COSMOS_MAX_INPUT_LEN` and `COSMOS_MAX_KV_CAPACITY` used during building (defaults 1024 and 2048), plus any explicit `COSMOS_MAX_IMAGE_TOKENS` and `COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE` overrides. Empty or absent visual overrides retain the upstream default profile identity. Keep the backend on loopback port 8000, which is the UI proxy's fixed default.

On the target, `sudo bash scripts/install_services.sh` checks that the launch dependencies exist, backs up existing unit files, verifies the replacement units, then installs and enables `cosmos-edge-backend.service` and `cosmos-edge-ui.service`. It does not start them. Stop the task's foreground servers before running `sudo systemctl start cosmos-edge-backend.service cosmos-edge-ui.service`; inspect startup with `systemctl status` or `journalctl -u cosmos-edge-backend -u cosmos-edge-ui`. The UI service binds to `127.0.0.1:8090` and needs an SSH tunnel for access from another computer. Backend startup validates the selected cache and does not intentionally build missing engines.

Review the generated manifest before transfer. After the target is supplied and SSH keys are provisioned, the coordinating agent can run:

```sh
python3 scripts/deploy_transfer.py send \
  --host TARGET_HOST_OR_SCOPED_IPV6 \
  --port 22 \
  --known-hosts /absolute/path/to/task-known-hosts \
  --identity-file /absolute/path/to/task-private-key
```

Sending requires the account `jetson`, an actual home directory `/home/jetson`, Python 3 and Git on the target. It uses only the supplied identity and pinned known-hosts file, disables SSH configuration/agent/password fallback, and rejects unknown or changed host keys. The target path is fixed to **`/home/jetson/cosmos-edge`**. The helper refuses an existing destination and never merges, overwrites, sudo-installs or deletes an existing project. If transfer fails, partial files remain inside that dedicated directory with an incomplete marker for inspection; the operator must resolve that state before retrying.

Both sides verify file sizes and SHA-256; the target additionally checks actual Git HEAD and submodule revisions before writing `DEPLOYMENT-VERIFIED.json`. The tar receiver allows only expected regular files, rejects path traversal, links and special files, and checks available disk space before creating the destination. The resulting project contains the backend under `external/TensorRT-Edge-LLM` and model under `models/cosmos3-edge-reasoner`. Installation, engine building, UI launch and actual model validation remain separate tasks. Rerun `prepare` after editing any allowlisted source file; `send` refuses stale preparation hashes.
