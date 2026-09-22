#!/usr/bin/env python3
"""Create device-local paths without changing the frozen deployment's tuning.

The private _validate command is shared by the systemd wrappers. It parses environment files as
literal data, never shell code, and returns only validated tab-separated fields.
"""
from __future__ import annotations

import argparse
import grp
import ipaddress
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys
import tempfile

PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_KEYS = {"COSMOS_MODEL_DIR", "COSMOS_CACHE_DIR", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"}
INTEGER_KEYS = {"COSMOS_MAX_INPUT_LEN", "COSMOS_MAX_KV_CAPACITY", "COSMOS_ENCODER_CACHE_BYTES",
                "COSMOS_MAX_IMAGE_TOKENS", "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE",
                "COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE"}
BOOLEAN_KEYS = {"COSMOS_STATIC_CLOCKS", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "PYTHONUNBUFFERED"}
ALLOWED_KEYS = PATH_KEYS | INTEGER_KEYS | BOOLEAN_KEYS | {"COSMOS_PROFILE", "COSMOS_BACKEND_ID", "COSMOS_TOP_P"}
SAFE_PATH = re.compile(r"/[A-Za-z0-9_./+\-]+\Z")
SAFE_ACCOUNT = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*\Z")


def safe_path(value: str | Path, label: str) -> Path:
    text = str(value)
    if not SAFE_PATH.fullmatch(text):
        raise ValueError(f"{label} must be an absolute path using only letters, digits, /, _, ., +, and -; whitespace, % and shell syntax are unsupported")
    resolved = Path(text).resolve()
    if not SAFE_PATH.fullmatch(str(resolved)):
        raise ValueError(f"{label} resolves to a path containing unsupported characters")
    return resolved


def parse_env(path: Path) -> dict[str, str]:
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        if line.startswith("COSMOS_STATIC_CLOCKS=") and line not in ("COSMOS_STATIC_CLOCKS=0", "COSMOS_STATIC_CLOCKS=1"):
            raise ValueError("Use the exact unquoted line COSMOS_STATIC_CLOCKS=0 or COSMOS_STATIC_CLOCKS=1")
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=([^\s\"'`$;\\]*)", line)
        if not match or match[1] not in ALLOWED_KEYS:
            raise ValueError(f"{path.name}:{number}: expected an allowed, unquoted KEY=value literal; shell statements are not supported")
        key, value = match.groups()
        if key in values:
            raise ValueError(f"Duplicate {key} assignments are not supported")
        values[key] = value
    return values


def validate_values(values: dict[str, str], *, require_model: bool = True) -> None:
    for key in ("COSMOS_PROFILE", "COSMOS_MODEL_DIR", "COSMOS_CACHE_DIR"):
        if not values.get(key):
            raise ValueError(f"{key} is required")
    if values["COSMOS_PROFILE"] not in ("fp16", "rtn-v1"):
        raise ValueError("COSMOS_PROFILE must be fp16 or rtn-v1")
    for key in PATH_KEYS & values.keys():
        path = safe_path(values[key], key)
        if path.exists() and not path.is_dir():
            raise ValueError(f"{key} must name a directory")
        if require_model and key == "COSMOS_MODEL_DIR" and not path.is_dir():
            raise ValueError("COSMOS_MODEL_DIR must name an existing local model directory")
    for key in INTEGER_KEYS & values.keys():
        if values["COSMOS_PROFILE"] == "fp16" and not values[key] and key in {
            "COSMOS_MAX_IMAGE_TOKENS", "COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE"
        }:
            continue  # Historical FP16 builds used the builder's unset visual overrides.
        minimum = 0 if key == "COSMOS_ENCODER_CACHE_BYTES" else 1
        if not re.fullmatch(r"[0-9]+", values[key]) or int(values[key]) < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in BOOLEAN_KEYS & values.keys():
        if values[key] not in ("0", "1"):
            raise ValueError(f"{key} must be 0 or 1")
    if "COSMOS_TOP_P" in values:
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", values["COSMOS_TOP_P"]) or not 0 < float(values["COSMOS_TOP_P"]) <= 1:
            raise ValueError("COSMOS_TOP_P must be a decimal in (0, 1]")
    if "COSMOS_BACKEND_ID" in values and not re.fullmatch(r"[A-Za-z0-9_.-]+", values["COSMOS_BACKEND_ID"]):
        raise ValueError("COSMOS_BACKEND_ID must contain only letters, digits, _, . or -")


def service_account(name: str | None) -> tuple[str, str]:
    if not name or not SAFE_ACCOUNT.fullmatch(name):
        raise ValueError("Provide --user EXISTING_NONROOT_ACCOUNT (or invoke through sudo with SUDO_USER set)")
    try:
        entry = pwd.getpwnam(name)
        group = grp.getgrgid(entry.pw_gid).gr_name
    except KeyError as exc:
        raise ValueError(f"Service account or primary group does not exist: {name}") from exc
    if entry.pw_uid == 0:
        raise ValueError("The service account must not be root")
    if not SAFE_ACCOUNT.fullmatch(group):
        raise ValueError("Service primary group contains unsupported systemd characters")
    return entry.pw_name, group


def local_ipv4(value: str) -> str:
    address = ipaddress.IPv4Address(value)
    if address.is_loopback or address.is_unspecified or address.is_multicast or address.is_link_local or int(address) == 0xffffffff:
        raise ValueError("Provide the device's LAN IPv4 address")
    ip_command = shutil.which("ip")
    if not ip_command:
        raise ValueError("The ip command is required to verify the device's actual IPv4 address")
    import json
    interfaces = json.loads(subprocess.run([ip_command, "-j", "-4", "address", "show"],
                                          check=True, capture_output=True, text=True).stdout)
    assigned = {item.get("local") for interface in interfaces for item in interface.get("addr_info", [])}
    if str(address) not in assigned:
        raise ValueError(f"{address} is not assigned to a local network interface")
    return str(address)


def check_service_access(project: Path, account: str, values: dict[str, str]) -> None:
    """Probe permissions as the service UID without writing anything or running project code."""
    entry = pwd.getpwnam(account)
    paths = [(project, os.R_OK | os.X_OK),
             (safe_path(values["COSMOS_MODEL_DIR"], "Model directory"), os.R_OK | os.X_OK)]
    for key in ("COSMOS_CACHE_DIR", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
        if key in values:
            paths.append((safe_path(values[key], key), os.R_OK | os.W_OK | os.X_OK))
    # These runtime outputs remain project-relative even with a custom --data-dir.
    for output_dir in (project / "results", project / "data/logs"):
        parent = output_dir
        while not parent.exists():
            parent = parent.parent
        paths.append((parent, os.R_OK | os.W_OK | os.X_OK))
    request_log = project / "data/logs/native-requests.jsonl"
    if request_log.exists():
        paths.append((request_log, os.W_OK))
    paths.append((project / "external/TensorRT-Edge-LLM/.venv/bin/python", os.R_OK | os.X_OK))

    def check() -> str:
        for path, mode in paths:
            if not os.access(path, mode):
                return f"Service user {account} lacks required access to {path}; configure paths as that user first"
        return ""

    if os.geteuid() != 0:
        if os.geteuid() != entry.pw_uid:
            raise ValueError("Validating a different service account requires sudo; --dry-run does not change files")
        error = check()
    else:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                os.initgroups(account, entry.pw_gid)
                os.setgid(entry.pw_gid)
                os.setuid(entry.pw_uid)
                error = check()
            except (OSError, ValueError) as exc:
                error = str(exc)
            os.write(write_fd, error.encode()[:4096])
            os.close(write_fd)
            os._exit(1 if error else 0)
        os.close(write_fd)
        try:
            error = os.read(read_fd, 4096).decode()
        finally:
            os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        if status and not error:
            error = "Could not verify the service account's directory permissions"
    if error:
        raise ValueError(error)


def validated_install(project: Path, user: str | None, env_file: str | None = None,
                      lan_ip: str | None = None) -> tuple[str, str, Path, str, str]:
    project = safe_path(project, "Project directory")
    account, group = service_account(user)
    default = project / "deployment/local.env"
    if not default.exists():
        default = project / "deployment/selected.env"
    selected = safe_path(env_file or default, "Environment file")
    if not selected.is_file():
        raise ValueError("Run configure_deployment.py first, or provide an existing --env-file")
    values = parse_env(selected)
    validate_values(values)
    check_service_access(project, account, values)
    return account, group, selected, values.get("COSMOS_STATIC_CLOCKS", "0"), local_ipv4(lan_ip) if lan_ip else ""


def configure(project: Path, model_dir: str, cache_dir: str, data_dir: str | None = None,
              *, force: bool = False) -> Path:
    if os.geteuid() == 0:
        raise ValueError("Run configure_deployment.py as the normal deployment user, without sudo")
    project = safe_path(project, "Project directory")
    model = safe_path(model_dir, "Model directory")
    cache = safe_path(cache_dir, "Engine cache directory")
    data = safe_path(data_dir or project / "data", "Data directory")
    target = project / "deployment/local.env"
    if target.exists() and not force:
        raise ValueError("deployment/local.env already exists; inspect it and use --force to replace it")
    values = parse_env(project / "deployment/selected.env")
    values.update(COSMOS_MODEL_DIR=str(model), COSMOS_CACHE_DIR=str(cache), TMPDIR=str(data / "tmp"),
                  XDG_CACHE_HOME=str(data / "cache"), CUDA_CACHE_PATH=str(data / "cache/cuda"))
    validate_values(values)
    for directory in (cache, data / "tmp", data / "cache/cuda"):
        directory.mkdir(parents=True, exist_ok=True)
    content = "# Device-local paths; frozen deployment/selected.env remains provenance.\n"
    content += "".join(f"{key}={value}\n" for key, value in values.items())
    fd, temporary = tempfile.mkstemp(prefix=".local.env.", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(content)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_validate":
        parser = argparse.ArgumentParser(description="Internal systemd wrapper validation")
        parser.add_argument("--project-dir", required=True, type=Path)
        parser.add_argument("--user", default=os.environ.get("SUDO_USER"))
        parser.add_argument("--env-file")
        parser.add_argument("--lan-ip")
        args = parser.parse_args(argv[1:])
        try:
            print("\t".join(map(str, validated_install(args.project_dir, args.user, args.env_file, args.lan_ip))))
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            parser.exit(2, f"Configuration error: {exc}\n")
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model-dir", required=True, help="Existing local model directory")
    parser.add_argument("--cache-dir", required=True, help="Local TensorRT engine cache directory")
    parser.add_argument("--data-dir", help="Writable temporary/cache parent (default: PROJECT/data)")
    parser.add_argument("--force", action="store_true", help="Replace an existing deployment/local.env")
    args = parser.parse_args(argv)
    try:
        path = configure(PROJECT_DIR, args.model_dir, args.cache_dir, args.data_dir, force=args.force)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    print(f"Wrote {path}; no engines, services, clocks or power settings were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
