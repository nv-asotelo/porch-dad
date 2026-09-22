#!/usr/bin/env python3
"""Prepare an allowlisted deployment and optionally stream it to a pinned SSH host.

Preparation is entirely local. Sending requires an explicit host and SSH identity;
it creates only /home/jetson/cosmos-edge and refuses an existing destination.

The model allowlist still contains the original verified FP16 reasoner checkpoint,
not the separately generated MLP INT4 weights or any built engine. With the MLP
profile selected, plain `prepare` refuses to imply a runnable deployment. Explicit
`prepare --source-stage` packages the source and original checkpoint as a staging
input, marks the missing selected artifacts in its manifest, and requires separate
MLP conversion, native build, engine build and validation before service startup.
The local public backend must match all six reviewed patches exactly; preparation
does not repair or change that checkout. See --help before using historical
three-patch transfer instructions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/deployment"
DESTINATION = "/home/jetson/cosmos-edge"
BACKEND = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"
MODEL = "344d602b128d1bbdacb43b08d0a3626f46343e29"
MODEL_DIRECTORY = "models/cosmos3-edge-reasoner"
ARCHIVE_SOURCE = "data/deployment/backend-checkout.tar"
ARCHIVE_TARGET = ".transfer/backend-checkout.tar"
REPOSITORIES = [
    ("", BACKEND, "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"),
    ("3rdParty/NVTX", "2fb879e512ed208f83c6aa4d4c96b958789edf49", "https://github.com/NVIDIA/NVTX.git"),
    ("3rdParty/googletest", "f132c893119698e10daef8525d0ad7a3f05176f2", "https://github.com/google/googletest.git"),
    ("3rdParty/nlohmannJson", "55f93686c01528224f448c19128836e7df245f72", "https://github.com/nlohmann/json.git"),
]
BACKEND_PATCHES = [
    "patches/cosmos3-patch-embedding-chw.patch",
    "patches/cosmos3-half-pixel-position.patch",
    "patches/tensorrt-edge-llm-v0.10.1-encoder-cache-budget.patch",
    "patches/int4-gemv-cosmos-mlp-n4.patch",
    "patches/cosmos-runtime-image-token-budget.patch",
    "patches/cosmos-encoder-cache-bypass.patch",
]
DIFF_ARGUMENTS = ("diff", "--binary", "--full-index", "--no-color", "--no-renames",
                  "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--")
ORIGINAL_FILES = [
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "GOAL.md", "sources.lock.json",
    "results/model-download.json",
    "deployment/selected.env", "deployment/selected-config.json",
    "scripts/serve_ui.py", "scripts/benchmark.py", "scripts/soak.py", "scripts/run_backend.sh", "scripts/fetch_sources.sh",
    "scripts/awq_isolated_env.sh", "scripts/quantize_cosmos3_awq.py", "scripts/device_inventory.sh",
    "scripts/install_jetpack_compute.sh", "scripts/build_backend.sh",
    "scripts/build_model_cache.py", "scripts/sample_storage_irqs.py", "scripts/measure_command.py",
    "scripts/repair_cosmos_runtime_config.py", "scripts/repair_cosmos_chat_template.py",
    "scripts/preflight_cosmos_artifacts.py", "scripts/serve_backend.py", "scripts/install_services.sh",
    "scripts/run_selected_backend.sh", "scripts/cosmos_runtime.py",
    "scripts/benchmark_native.py", "scripts/benchmark_ttft.py", "scripts/validate_runtime_controls.py", "scripts/compare_server_timings.py",
    "scripts/quantize_cosmos3_rtn.py", "scripts/rtn_backend.py", "scripts/run_quality_suite.py",
    "scripts/host_probe.py",
    "scripts/configure_deployment.py", "scripts/jetson_preflight.py", "scripts/fetch_reasoner.py",
    "scripts/enable_lan_ui.sh",
    "scripts/deploy_transfer.py", "scripts/ssh_nfs_bridge.py",
    "web/app.js", "web/index.html", "web/style.css",
    "tests/test_benchmark.py", "tests/test_soak.py", "tests/test_serve_ui.py", "tests/test_sse.js",
    "tests/test_runtime_config_repair.py", "tests/test_chat_template_repair.py",
    "tests/test_cosmos_preflight.py", "tests/test_encoder_cache_budget.py", "tests/test_deploy_transfer.py",
    "tests/test_cosmos3_patch_layout.py", "tests/test_cosmos3_position_layout.py",
    "tests/test_rtn_backend.py", "tests/test_rtn_quantization.py",
    "tests/test_selected_backend.py",
    "tests/test_visual_profile.py", "tests/test_cosmos_runtime.py", "tests/test_cosmos_runtime_routes.py",
    "tests/test_configure_deployment.py", "tests/test_jetson_preflight.py", "tests/test_fetch_reasoner.py",
    "tests/test_enable_lan_ui.py",
    "tests/test_runtime_image_budget_patch.py",
    "tests/test_compare_server_timings.py", "tests/test_ui_metrics.js", "tests/test_capture_presets.js",
    "patches/cosmos3-patch-embedding-chw.patch", "patches/cosmos3-patch-embedding-chw.md",
    "patches/cosmos3-half-pixel-position.patch", "patches/cosmos3-half-pixel-position.md",
    "patches/tensorrt-edge-llm-v0.10.1-encoder-cache-budget.patch", "patches/encoder-cache-budget.json",
    "patches/int4-gemv-cosmos-mlp-n4.patch", "patches/cosmos-runtime-image-token-budget.patch",
    "patches/cosmos-encoder-cache-bypass.patch",
    "docs/backend-build.md", "docs/expand-nvme-rootfs.md", "docs/jetpack-install.md",
    "docs/quality-workload.md", "docs/api-streaming-soak.md", "docs/usb-nfs-bridge.md", "docs/usb-package-proxy.md", "docs/deployment-transfer.md",
    "docs/agent-deployment.md",
    "research/benchmark-method.md", "research/backend-feasibility.md", "research/ui-feasibility.md",
    "research/backend-preflight-audit.md", "research/cosmos3-kernel-selection.md",
    "research/fp16-build-memory.md", "research/jetpack-compute-install.md",
    "research/transformers-registration-audit.md", "research/cosmos3-awq-candidate.md",
    "research/python-system-site-compatibility.md",
    "research/contribution-ledger.md", "research/nvme-timeout-investigation.md",
    "research/reference-quality-diagnostic.md",
    "research/runtime-controls-server-timing.md",
    "research/int4-without-aux-gpu.md", "research/calibration-data.md",
    "research/bsp-preflight-audit.md", "research/confirmed-board-flash-options.md",
    "research/initrd-nfs-ssh-tunnel-audit.md", "research/iso-recovery-assets.md",
    "research/macos-recovery-usb-publication.md", "research/prepared-super-nvme-package-review.md",
    "research/recovery-flash.md", "research/usb-first-upload-diagnosis.md", "research/public-metadata.json",
    "benchmarks/fixtures/generate.py", "benchmarks/fixtures/manifest.json",
    "benchmarks/fixtures/convert_jpeg.py", "benchmarks/fixtures/jpeg-manifest.json",
    "benchmarks/fixtures/jpeg/01-left-right.jpg", "benchmarks/fixtures/jpeg/02-counts.jpg",
    "benchmarks/fixtures/jpeg/03-above-below.jpg", "benchmarks/fixtures/jpeg/04-inside-outside.jpg",
    "benchmarks/fixtures/jpeg/05-relative-size.jpg", "benchmarks/fixtures/jpeg/06-quadrants.jpg",
    "benchmarks/fixtures/contact-sheet.png", "benchmarks/fixtures/01-left-right.png",
    "benchmarks/fixtures/02-counts.png", "benchmarks/fixtures/03-above-below.png",
    "benchmarks/fixtures/04-inside-outside.png", "benchmarks/fixtures/05-relative-size.png",
    "benchmarks/fixtures/06-quadrants.png",
    "benchmarks/live-vlm-1280/manifest.json", "benchmarks/live-vlm-1280/README.md",
    "benchmarks/live-vlm-1280/01-action-camera.jpg", "benchmarks/live-vlm-1280/02-pen.jpg",
    "benchmarks/live-vlm-1280/03-scissors.jpg",
    "benchmarks/natural-smoke/README.md", "benchmarks/natural-smoke/manifest.json",
    "benchmarks/natural-smoke/images/000000023781.jpg",
    "benchmarks/natural-smoke/images/000000027932.jpg",
    "benchmarks/natural-smoke/images/000000029393.jpg",
    "benchmarks/natural-smoke/metadata/coco-selected-entries.json",
    "benchmarks/natural-smoke/metadata/download-records.json",
    "benchmarks/natural-smoke/metadata/flickr-3669674438.json",
    "benchmarks/natural-smoke/metadata/flickr-837387952.json",
    "benchmarks/natural-smoke/metadata/flickr-4228514131.json",
]


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts) or "\\" in value:
        raise ValueError("Unsafe relative path: " + value)
    return path


def local_file(relative):
    relative_path(relative)
    path = ROOT / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("Expected a regular, in-project file without symlink components: " + relative)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(path, *arguments, raw=False):
    environment = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_TERMINAL_PROMPT="0", GIT_CONFIG_COUNT="0")
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        environment.pop(key, None)
    output = subprocess.check_output(["git", "-c", "protocol.file.allow=always", "-C", str(path), *arguments],
                                     env=environment, text=not raw, stderr=subprocess.PIPE)
    return output if raw else output.strip()


def verified_model_files():
    record = json.loads(local_file("results/model-download.json").read_text())
    if record.get("revision") != MODEL or record.get("complete") is not True or record.get("errors"):
        raise ValueError("The exact public reasoner download is not verified complete")
    model = ROOT / MODEL_DIRECTORY
    for forbidden in ("transformer/config.json", "vae/config.json"):
        if (model / forbidden).exists():
            raise ValueError("Reasoner transfer refuses generator component config: " + forbidden)
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "cosmos3_edge" or not config.get("vision_config"):
        raise ValueError("Expected the Cosmos3-Edge multimodal reasoner")
    files = {}
    for name, entry in record["files"].items():
        relative_path(name)
        if entry.get("verified") is not True or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError("Unverified model file: " + name)
        files[f"{MODEL_DIRECTORY}/{name}"] = entry
    shards = set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"].values())
    if any(f"{MODEL_DIRECTORY}/{name}" not in files for name in shards):
        raise ValueError("Reasoner index references an unverified shard")
    return files


def deployment_scope(*, source_stage=False):
    """Describe what is actually bundled without sourcing an environment file."""
    settings = {}
    for line in local_file("deployment/selected.env").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in settings:
            raise ValueError("Expected unique, unquoted NAME=value settings in deployment/selected.env")
        settings[key] = value
    profile = settings.get("COSMOS_PROFILE")
    selected_model = settings.get("COSMOS_MODEL_DIR", "")
    if profile not in {"fp16", "rtn-v1"} or not selected_model.startswith(DESTINATION + "/models/"):
        raise ValueError("Selected profile/model path is outside the reviewed deployment scope")
    relative_path(selected_model.removeprefix(DESTINATION + "/"))
    included_model = DESTINATION + "/" + MODEL_DIRECTORY
    selected_weights_included = profile == "fp16" and selected_model == included_model
    if not selected_weights_included and not source_stage:
        raise ValueError(
            "The selected " + str(profile) + " service requires " + selected_model
            + ", but this transfer includes only the verified original FP16 reasoner checkpoint at "
            + included_model + ". MLP INT4 weights and engines are not included. "
            "Use prepare --source-stage only to stage sources/checkpoint, then separately generate or "
            "verify the selected weights, build native code/engines, and validate before starting services.")
    followup = ["Build native runtime from the six recorded source patches.",
                "Build and validate the selected engine; engine caches and native binaries are not transferred.",
                "Validate runtime controls, image quality and local service readiness before starting services."]
    if not selected_weights_included:
        followup.insert(0, "Generate or separately transfer and verify the selected MLP INT4 weights; "
                           "only the original verified reasoner checkpoint is included.")
    return {"kind": "source_stage" if source_stage else "fp16_checkpoint_and_sources",
            "selected_profile": profile, "selected_model_path": selected_model,
            "included_model_path": included_model, "selected_weights_included": selected_weights_included,
            "engine_caches_included": False, "native_binaries_included": False,
            "ready_to_start_selected_service": False, "required_followup": followup}


def require_reviewed_backend_diff(actual, expected):
    if actual != expected:
        raise ValueError(
            "Backend tracked edits differ from the " + str(len(BACKEND_PATCHES))
            + " reviewed patches (" + ", ".join(Path(item).name for item in BACKEND_PATCHES)
            + "). Nothing was silently omitted. Reconstruct the same reviewed source state "
            "in the local pinned public checkout before preparing a transfer; this helper does not modify it.")


def backend_snapshot(destination):
    upstream = ROOT / "external/TensorRT-Edge-LLM"
    for relative, revision, _ in REPOSITORIES:
        checkout = upstream / relative
        if git(checkout, "rev-parse", "HEAD") != revision or (relative and git(checkout, "status", "--porcelain", "--untracked-files=no")):
            raise ValueError("Pinned backend/submodule checkout is missing, modified or at another revision: " + relative)
        if relative and git(upstream, "ls-tree", "HEAD", relative).split()[2] != revision:
            raise ValueError("Submodule pin disagrees with the parent gitlink: " + relative)
    with tempfile.TemporaryDirectory(prefix="backend-stage-", dir=PACKAGE) as temporary:
        stage = Path(temporary) / "TensorRT-Edge-LLM"
        empty_template = Path(temporary) / "empty-template"
        empty_template.mkdir()
        for relative, revision, url in REPOSITORIES:
            target = stage / relative
            if target.exists():
                target.rmdir()  # Only the fresh clone's empty submodule placeholder.
            git(Path(temporary), "clone", "--no-local", "--depth", "1", "--no-checkout",
                "--template=" + str(empty_template), (upstream / relative).resolve().as_uri(), str(target))
            git(target, "checkout", "--detach", revision)
            git(target, "remote", "set-url", "origin", url)
            for name in ("logs", "FETCH_HEAD", "ORIG_HEAD"):
                path = target / ".git" / name
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            if git(target, "rev-parse", "HEAD") != revision:
                raise ValueError("Sanitized clone did not preserve the required commit")
        git(stage, "submodule", "init")
        if any(line[0] != " " for line in subprocess.check_output(
                ["git", "-C", str(stage), "submodule", "status", "--recursive"], text=True).splitlines()):
            raise ValueError("Sanitized submodules are not initialized at their exact pins")
        # Reconstruct only the reviewed original patches on a clean public pin.
        # Comparing the complete final diff rejects omitted or undocumented edits.
        patches = []
        for relative in BACKEND_PATCHES:
            patch = local_file(relative)
            patches.append({"path": relative, "sha256": sha256(patch), "bytes": patch.stat().st_size})
            git(stage, "apply", "--check", str(patch))
            git(stage, "apply", str(patch))
        expected_diff = git(stage, *DIFF_ARGUMENTS, raw=True)
        require_reviewed_backend_diff(git(upstream, *DIFF_ARGUMENTS, raw=True), expected_diff)
        changed_files = []
        for relative in git(stage, "diff", "--name-only", "HEAD").splitlines():
            relative_path(relative)
            patched = stage / relative
            source = upstream / relative
            if not patched.is_file() or patched.is_symlink() or not source.is_file() or source.is_symlink():
                raise ValueError("Documented backend patches must modify existing regular source files")
            before = git(stage, "show", "HEAD:" + relative, raw=True)
            if sha256(source) != sha256(patched):
                raise ValueError("Patched source changed during snapshot: " + relative)
            changed_files.append({"path": relative, "before_sha256": hashlib.sha256(before).hexdigest(),
                                  "after_sha256": sha256(patched), "bytes": patched.stat().st_size})
        source_state = {"source_revision": BACKEND, "patches": patches, "patched_files": changed_files,
                        "tracked_diff_sha256": hashlib.sha256(expected_diff).hexdigest(),
                        "tracked_diff_bytes": len(expected_diff),
                        "diff_arguments": list(DIFF_ARGUMENTS)}
        with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_dir():
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Backend snapshot refuses links and special files")
                info = archive.gettarinfo(str(path), arcname=str(path.relative_to(stage)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
        if git(upstream, *DIFF_ARGUMENTS, raw=True) != expected_diff or any(
                sha256(local_file(item["path"])) != item["sha256"] for item in patches):
            raise ValueError("Backend source or documented patches changed while creating the archive")
        return source_state


def prepare(*, source_stage=False):
    scope = deployment_scope(source_stage=source_stage)
    if PACKAGE.resolve() != PACKAGE:
        raise ValueError("Deployment package directory must remain inside the actual project data directory")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    model_files = verified_model_files()
    files = []
    for relative in ORIGINAL_FILES + sorted(model_files):
        path = local_file(relative)
        digest = sha256(path)
        expected = model_files.get(relative)
        if expected and (digest != expected["sha256"] or path.stat().st_size != expected["size_bytes"]):
            raise ValueError("Verified model bytes changed: " + relative)
        files.append({"source": relative, "target": relative, "sha256": digest,
                      "bytes": path.stat().st_size, "mode": 0o755 if os.access(path, os.X_OK) else 0o644})
    archive = ROOT / ARCHIVE_SOURCE
    pending = archive.with_suffix(".tar.pending")
    try:
        backend_state = backend_snapshot(pending)
        pending.replace(archive)
    finally:
        pending.unlink(missing_ok=True)
    files.append({"source": ARCHIVE_SOURCE, "target": ARCHIVE_TARGET, "sha256": sha256(archive),
                  "bytes": archive.stat().st_size, "mode": 0o644})
    manifest = {
        "schema_version": 1, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "destination": DESTINATION, "status": "prepared_locally_not_transferred",
        "deployment_scope": scope,
        "backend": {"path": "external/TensorRT-Edge-LLM", "revision": BACKEND,
                    "source_state": backend_state,
                    "repositories": [{"path": path, "revision": revision, "public_url": url}
                                     for path, revision, url in REPOSITORIES]},
        "model": {"path": MODEL_DIRECTORY, "revision": MODEL, "repository": "https://huggingface.co/nvidia/Cosmos3-Edge",
                  "license": "OpenMDW-1.1", "source_verification": "results/model-download.json; every selected file SHA-256 rechecked",
                  "inference_validated": False},
        "exclusions": [".qa and credentials", "original Git configuration/hooks/history outside pinned backend commits",
                       "other external repositories", "downloads", "build/engine caches", "virtual environments",
                       "unverified model files", "generated MLP INT4 model weights", "unlisted project files"],
        "files": files, "file_count": len(files), "transfer_bytes": sum(item["bytes"] for item in files),
        "note": "Backend archive contains sanitized Git repositories preserving exact HEAD/submodule pins plus all six reviewed patches, with patch/file/full-diff hashes. Model scope is the original verified FP16 checkpoint only; inspect deployment_scope for missing selected artifacts. No target access, package installation, build or inference is performed during preparation.",
    }
    output = PACKAGE / "manifest.json"
    temporary = output.with_suffix(".json.pending")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"manifest": str(output), "files": len(files), "transfer_bytes": manifest["transfer_bytes"],
                      "status": manifest["status"], "deployment_scope": scope}, indent=2))


REMOTE_RECEIVER = r'''
import hashlib, json, os, pathlib, shutil, subprocess, sys, tarfile
ROOT = pathlib.Path('/home/jetson/cosmos-edge')
if shutil.which('git') is None:
    raise SystemExit('Git is required on the target; bootstrap it separately before transferring files')
if pathlib.Path.home() != ROOT.parent or ROOT.parent.is_symlink() or ROOT.parent.resolve() != ROOT.parent:
    raise SystemExit('Run as the target account whose real home is /home/jetson')
if ROOT.exists() or ROOT.is_symlink():
    raise SystemExit('Destination already exists; refusing to merge or overwrite it')
def relative(name):
    p = pathlib.PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe archive path')
    return p
def extract_regular(archive, member, base):
    if not member.isfile():
        raise ValueError('Only regular files are accepted')
    target = base.joinpath(*relative(member.name).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as output:
        source = archive.extractfile(member)
        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    target.chmod(0o755 if member.mode & 0o111 else 0o644)
    return target
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as archive:
    first = archive.next()
    if first is None or first.name != 'deployment-manifest.json' or not first.isfile() or first.size > 2000000:
        raise SystemExit('Missing bounded deployment manifest')
    raw = archive.extractfile(first).read()
    manifest = json.loads(raw)
    if manifest.get('destination') != str(ROOT) or manifest.get('schema_version') != 1:
        raise SystemExit('Wrong deployment destination/schema')
    required = manifest['transfer_bytes'] + sum(item['bytes'] for item in manifest['files']
        if item['target'] == '.transfer/backend-checkout.tar') + 1024 ** 3
    if shutil.disk_usage(ROOT.parent).free < required:
        raise SystemExit('Insufficient free space for transfer, backend expansion and safety margin')
    ROOT.mkdir(mode=0o750)
    (ROOT/'deployment-manifest.json').write_bytes(raw)
    (ROOT/'.transfer').mkdir()
    incomplete = ROOT/'.transfer/INCOMPLETE'
    incomplete.write_text('Transfer is incomplete until DEPLOYMENT-VERIFIED.json exists.\n')
    expected = {item['target']: item for item in manifest['files']}
    if len(expected) != len(manifest['files']): raise ValueError('Duplicate manifest targets')
    received = set()
    for member in iter(archive.next, None):
        if member.name not in expected or member.name in received:
            raise ValueError('Unexpected or duplicate transfer member')
        item = expected[member.name]
        if member.size != item['bytes']: raise ValueError('Transferred size mismatch')
        path = extract_regular(archive, member, ROOT)
        if digest(path) != item['sha256']: raise ValueError('Transferred SHA-256 mismatch: '+member.name)
        received.add(member.name)
    if received != set(expected): raise ValueError('Incomplete transfer')
backend = ROOT/'external/TensorRT-Edge-LLM'
backend.mkdir(parents=True)
with tarfile.open(ROOT/'.transfer/backend-checkout.tar', 'r') as archive:
    for member in archive:
        extract_regular(archive, member, backend)
for item in manifest['backend']['repositories']:
    checkout = backend/item['path']
    # Regular-file archives omit empty refs in detached shallow clones. Git needs
    # this directory to recognize each repository instead of finding its parent.
    (checkout/'.git/refs').mkdir(exist_ok=True)
    revision = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'], text=True).strip()
    if revision != item['revision']: raise ValueError('Backend/submodule revision mismatch')
status = subprocess.check_output(['git','-C',str(backend),'submodule','status','--recursive'],text=True)
if any(line[0] != ' ' for line in status.splitlines()): raise ValueError('Submodule state mismatch')
state = manifest['backend'].get('source_state')
if state:
    if state['source_revision'] != manifest['backend']['revision']:
        raise ValueError('Patched source revision mismatch')
    for item in state['patched_files']:
        if digest(backend.joinpath(*relative(item['path']).parts)) != item['after_sha256']:
            raise ValueError('Patched backend file SHA-256 mismatch: '+item['path'])
    diff_arguments = ['diff','--binary','--full-index','--no-color','--no-renames',
        '--no-ext-diff','--no-textconv','--src-prefix=a/','--dst-prefix=b/','HEAD','--']
    if state['diff_arguments'] != diff_arguments:
        raise ValueError('Unexpected backend diff verification arguments')
    diff = subprocess.check_output(['git','-C',str(backend), *diff_arguments])
    if len(diff) != state['tracked_diff_bytes'] or hashlib.sha256(diff).hexdigest() != state['tracked_diff_sha256']:
        raise ValueError('Patched backend tracked diff mismatch')
incomplete.unlink()
(ROOT/'DEPLOYMENT-VERIFIED.json').write_text(json.dumps({'status':'files_verified','file_count':len(received),
    'backend_revision':manifest['backend']['revision'],'model_revision':manifest['model']['revision'],
    'backend_source_state':state,
    'deployment_scope':manifest.get('deployment_scope'),
    'package_installation_performed':False,'inference_validated':False},indent=2)+'\n')
print('Deployment files and exact Git revisions verified under '+str(ROOT),flush=True)
'''


def private_file(value, secret):
    path = Path(value).expanduser()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & (0o077 if secret else 0o022):
        raise ValueError("SSH identity must be owner-only; host-key file must not be group/other writable; neither may be a symlink")
    return path.resolve()


def send(args):
    if not re.fullmatch(r"[A-Za-z0-9_.:%-]+", args.host) or args.host.startswith("-") or not 1 <= args.port <= 65535:
        raise ValueError("Supply a plain SSH hostname/scoped IP and valid port")
    key = private_file(args.identity_file, True)
    hosts = private_file(args.known_hosts, False)
    manifest_path = PACKAGE / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    recorded_scope = manifest.get("deployment_scope", {})
    expected_scope = deployment_scope(source_stage=recorded_scope.get("kind") == "source_stage")
    if recorded_scope != expected_scope:
        raise ValueError("Manifest lacks the current selected-artifact staging scope; rerun prepare")
    allowed = set(ORIGINAL_FILES) | set(verified_model_files()) | {ARCHIVE_SOURCE}
    if manifest.get("destination") != DESTINATION or {item["source"] for item in manifest["files"]} != allowed:
        raise ValueError("Manifest differs from the explicit deployment allowlist; rerun prepare")
    source_state = manifest.get("backend", {}).get("source_state", {})
    if (source_state.get("source_revision") != BACKEND
            or [item["path"] for item in source_state.get("patches", [])] != BACKEND_PATCHES
            or source_state.get("diff_arguments") != list(DIFF_ARGUMENTS)):
        raise ValueError("Manifest lacks the documented patched-backend provenance; rerun prepare")
    upstream = ROOT / "external/TensorRT-Edge-LLM"
    if (git(upstream, "rev-parse", "HEAD") != BACKEND
            or hashlib.sha256(git(upstream, *DIFF_ARGUMENTS, raw=True)).hexdigest() != source_state["tracked_diff_sha256"]):
        raise ValueError("Backend tracked source changed since preparation; rerun prepare")
    for relative, revision, _ in REPOSITORIES[1:]:
        checkout = upstream / relative
        if (git(checkout, "rev-parse", "HEAD") != revision
                or git(checkout, "status", "--porcelain", "--untracked-files=no")):
            raise ValueError("Pinned submodule changed since preparation; rerun prepare: " + relative)
    for item in manifest["files"]:
        expected_target = ARCHIVE_TARGET if item["source"] == ARCHIVE_SOURCE else item["source"]
        path = local_file(item["source"])
        if item["target"] != expected_target or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Prepared file changed; rerun prepare: " + item["source"])
    command = ["ssh", "-F", os.devnull, "-T", "-p", str(args.port), "-i", str(key),
               "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
               "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(hosts),
               "-o", "GlobalKnownHostsFile=" + os.devnull, "-o", "ConnectTimeout=10",
               "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no",
               "jetson@" + args.host, shlex.join(["python3", "-c", REMOTE_RECEIVER])]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            info = tarfile.TarInfo("deployment-manifest.json")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
            for item in manifest["files"]:
                path = local_file(item["source"])
                info = archive.gettarinfo(str(path), arcname=item["target"])
                info.mode = item["mode"]
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Target verification failed; any partial data remains confined to the dedicated project directory")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser("prepare", help="Locally verify files and create an explicitly scoped deployment manifest")
    preparation.add_argument("--source-stage", action="store_true",
        help="Explicitly stage source plus original FP16 checkpoint when selected MLP weights/engines are absent; not service-ready")
    transfer = commands.add_parser("send", help="Explicitly transfer a prepared package into a new dedicated Jetson project")
    transfer.add_argument("--host", required=True)
    transfer.add_argument("--port", type=int, default=22)
    transfer.add_argument("--known-hosts", required=True)
    transfer.add_argument("--identity-file", required=True)
    args = parser.parse_args()
    try:
        prepare(source_stage=args.source_stage) if args.action == "prepare" else send(args)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("Deployment preparation/transfer failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
