"""Canonical brockone serving entry point; unchanged archived inference backend."""
import argparse
import hashlib
import importlib.util
import ipaddress
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/jetson/brockone')
HERE = Path(__file__).absolute().parent
SOURCE_MANIFEST_SHA = 'e4dc6e7fbb66ce785adbceec602b950708ddc197b2c160f1556b99f1feb85cb5'
SERVING_SHA = '7fdcc1daa8a0e470a6bf14c24e307b48c680d094222abfd1612b598985e2c360'
CANDIDATE_ASSETS_SHA = '4bb54fc62a71f42d27f56d016c0ea828171a966c16c7617b50b424a19d50ff55'
PARENT_SHA = 'dbace8556c400bb0669ecbda6e139e54695ca9a6453dcde3f00ead8a65de7b53'
VALIDATOR_SHA = 'c16035bd990ed391417de8e73065c93ccf67b8cdf8a93278ba257b2d698f7290'
SELECTED72_SHA = 'cfb1152d26e51b62befa3ba676e8f260052da4f7141c05307abfc1fc373fbacb'
SELECTED72_REVIEW_SHA = '8ca6cb96bad2f15c0716ee125a08ab486e722deda864402adf90284bd9b051b3'
EDGE_COMMIT = 'e8b29522938901f6df19ebeedd4b69bc8edbcd97'
CANDIDATES = {
    'fp16': ('04c5962b4a4037382e5bc31059e1dfdb3451c32f199d911d3f113f9f9fe801fb', 0),
    'rtn169': ('f33cdc943e09558a96b52b5c9b22bef69c07cb9a89f15b7e088a6b2cc407a9e0', 169),
    'rtn168_head_fp16': ('ba9f41098b3ed7cdd65fb9bcdf66a901ce5a784ad514d29cb46fb0366c13afe4', 168),
    'awq_export_native': ('911c3c6ddd33ee52d0d30c2106b8dbde0f737ab5f76b08cb0fa7108e8005e835', 168),
}


def need(value, message):
    if not value:
        raise ValueError(message)


def safe(value):
    path = Path(value)
    need(path.is_absolute() and '..' not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), 'Unsafe path')
    return path


def owned(value):
    path = safe(value)
    need(path.is_relative_to(ROOT) and path != ROOT, 'Path must be inside separate SD brockone root')
    return path


def identity(path):
    path = safe(path)
    need(path.is_file(), 'Missing regular file')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def ref(row):
    path = owned(row['path'])
    need(identity(path) == {k: row[k] for k in ('sha256', 'bytes')}, 'Changed bound receipt')
    return json.loads(path.read_text())


def source_closure(root):
    root = owned(root)
    manifest = HERE / 'source-package-manifest.json'
    need(identity(manifest)['sha256'] == SOURCE_MANIFEST_SHA, 'Changed118 source manifest')
    expected = json.loads(manifest.read_text())['files']
    actual = {}
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'Source symlink')
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = {**identity(path), 'mode': path.stat().st_mode & 0o777}
    need(len(actual) == 118 and actual == expected, 'Original118 source closure differs')
    need(identity(HERE / 'brockone_serving.py')['sha256'] == SERVING_SHA, 'Unreviewed serving-only overlay')
    return actual


def overlay_closure(config):
    expected = config['overlay_files']
    need(set(expected) == {'brockone', 'brockone_launcher.py', 'brockone_serving.py'}, 'Exact naming launcher closure required')
    need({name: identity(HERE / name) for name in expected} == expected, 'Changed naming launcher bytes')


def validate_config(path, digest):
    path = owned(path)
    need(identity(path)['sha256'] == digest, 'Installation config SHA differs')
    config = json.loads(path.read_text())
    need(config['schema'] == 'brockone-install-config/v1' and config['model'] == 'brockone'
         and config['state'] == 'bound_target_build_for_qualification', 'Draft config is not executable')
    overlay_closure(config)
    need(config['candidate'] in CANDIDATES and config['image_tokens'] == 512 and config['max_new_tokens'] == 64,
         'Unselected candidate or changed fixed budgets')
    ipaddress.ip_address(config['host'])
    need(type(config['port']) is int and 1024 <= config['port'] <= 65535 and config['port'] not in {8091, 8443}, 'Unsafe or known-occupied service port')
    candidate_sha, linears = CANDIDATES[config['candidate']]
    need(config['candidate_receipt']['sha256'] == candidate_sha, 'Wrong selected candidate receipt')
    ref(config['candidate_receipt'])
    need(config['selected72_receipt']['sha256'] == SELECTED72_SHA
         and config['selected72_review']['sha256'] == SELECTED72_REVIEW_SHA, 'Exact reviewed selected72 result required')
    val = ref(config['selected72_receipt'])
    review = ref(config['selected72_review'])
    need(review['state'] == 'independently_replayed_complete_selected72_validation' and review['findings'] == []
         and review['terminal_receipt'] == {k: config['selected72_receipt'][k] for k in ('sha256', 'bytes')}, 'Selected72 independent review mismatch')
    need(val['schema'] == 'brockone-selected72-validation/v1' and val['state'] == 'complete_selected72_validation'
         and val['operator']['sha256'] == VALIDATOR_SHA and val['inputs_verified_after'] is True
         and val['operator_verified_after'] is True and val['same_device'] is True
         and val['no_final_cohorts_opened'] is True and val['original_adapter_parity_passed'] is False,
         'Actual completed selected72 receipt required')
    need([r['name'] for r in val['runs']] == ['bf16', *CANDIDATES]
         and all(r['state'] == 'complete' and r['all72_raw_and_prepared_inputs_verified'] is True for r in val['runs']), 'Incomplete selected72 comparison')
    paths = {k: owned(config[k]) for k in ('source_root', 'engine_dir', 'onnx_llm_dir', 'bindings_dir', 'plugin')}
    build = ref(config['engine_build_receipt'])
    runtime = ref(config['runtime_build_receipt'])
    need(Path(config['engine_build_receipt']['path']) == paths['engine_dir'] / 'brock-two-engine-build.json', 'Wrong engine receipt placement')
    need(build['measurement_scope'] == 'target_device' and build['int4_plugin_count'] == linears
         and build['unquantized_checkpoint_identity']['sha256'] == PARENT_SHA
         and build['runtime_build_manifest_sha256'] == config['runtime_build_receipt']['sha256'],
         'Wrong target engine/selected parent')
    assets_path = HERE / 'candidate-assets.json'
    need(identity(assets_path)['sha256'] == CANDIDATE_ASSETS_SHA, 'Changed executed candidate catalog')
    expected_assets = json.loads(assets_path.read_text())[config['candidate']]
    need(all(build[k] == v for k, v in expected_assets.items()), 'Target build did not consume the exact selected candidate/ONNX')
    need(runtime['target'] == 'orin' and runtime['gpu_sm'] == 87 and runtime['build_completed'] is True
         and runtime['measurement_scope'] == 'target_device' and runtime['source_revision'] == EDGE_COMMIT
         and 'Orin' in runtime['device_model'] and runtime['nv_tegra_release'], 'Actual Orin runtime build required')
    target_runtime_paths(config, paths, runtime, build)
    sources = source_closure(paths['source_root'])
    return config, paths, sources


def target_runtime_paths(config, paths, runtime, build):
    """Exact observed runtime05 aliases; config continues to name regular files."""
    base = owned(Path(config['runtime_build_receipt']['path']).parent)
    regular = base / 'libNvInfer_edgellm_plugin.so.1.0'
    binding = base / 'pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so'
    need(paths['plugin'] == regular and paths['bindings_dir'] == binding.parent
         and runtime['plugin'] == str(base / 'libNvInfer_edgellm_plugin.so')
         and runtime['runtime_binding'] == str(binding), 'Target runtime paths differ')
    expected_plugin = {'sha256': 'f4f15514300fdffa18a110f7708f6a816792bd6467809348b20c15e895ecaec7', 'bytes': 19121864}
    expected_binding = {'sha256': '993636c6ad2d33ff3c3a63ab3c770b9c445360cd958f3465f1cac26c63df51be', 'bytes': 24906392}
    need(identity(regular) == expected_plugin
         and runtime['plugin_sha256'] == build['runtime_plugin_sha256'] == expected_plugin['sha256']
         and identity(binding) == expected_binding
         and runtime['runtime_binding_sha256'] == expected_binding['sha256'], 'Target runtime bytes differ')
    # These are the only symlinks admitted. The parent and both concrete files
    # have already passed the ordinary no-symlink owned()/identity() checks.
    for name, target in [('libNvInfer_edgellm_plugin.so', 'libNvInfer_edgellm_plugin.so.1'),
                         ('libNvInfer_edgellm_plugin.so.1', 'libNvInfer_edgellm_plugin.so.1.0')]:
        alias = base / name
        need(alias.is_symlink() and os.readlink(alias) == target
             and alias.resolve(strict=True) == regular, 'Unrecorded runtime plugin alias')


def imported_source_guard(root):
    for name, module in tuple(sys.modules.items()):
        if name == 'brock_two' or name.startswith('brock_two.'):
            path = safe(module.__file__)
            need(path.is_relative_to(root / 'src'), 'Inference import outside exact118 source')


def serve(config, paths, sources):
    need(platform.machine() in {'aarch64', 'arm64'}, 'Target Orin execution only')
    need('Orin' in Path('/proc/device-tree/model').read_text() and Path('/etc/nv_tegra_release').is_file(), 'Actual target Orin required')
    need(Path('/sys/block/mmcblk0/device/type').read_text().strip() == 'SD', 'Confirmed SD device required')
    for path in (ROOT, *paths.values()):
        need(subprocess.check_output(['findmnt', '-n', '-o', 'SOURCE', '--target', str(path)], text=True).strip() == '/dev/mmcblk0p1', 'Keep all new brockone inputs on the confirmed SD filesystem')
    import shutil
    limits = config['startup_admission']
    need(type(limits['minimum_available_ram_bytes']) is int and limits['minimum_available_ram_bytes'] >= 4 * 1024**3
         and type(limits['minimum_free_disk_bytes']) is int and limits['minimum_free_disk_bytes'] >= 4 * 1024**3, 'Conservative resource guard required')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    need(int(memory['MemAvailable'].split()[0]) * 1024 >= limits['minimum_available_ram_bytes'], 'Insufficient available shared RAM; do not stop another service')
    need(shutil.disk_usage(ROOT).free >= limits['minimum_free_disk_bytes'], 'SD free-space guard failed')
    sys.dont_write_bytecode = True
    imported_source_guard(paths['source_root'])
    sys.path.insert(0, str(paths['source_root'] / 'src'))
    spec = importlib.util.spec_from_file_location('brockone_serving', HERE / 'brockone_serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported_source_guard(paths['source_root'])
    args = SimpleNamespace(checkpoint=None, endpoint=None, edge_engine_dir=paths['engine_dir'],
                           edge_onnx_llm_dir=paths['onnx_llm_dir'], edge_bindings_dir=paths['bindings_dir'],
                           edge_plugin=paths['plugin'], processor=None, layout='native', device='cuda',
                           image_tokens=512, max_new_tokens=64, profile=False, host=config['host'], port=config['port'])
    backend = module.backend_from_args(args)
    imported_source_guard(paths['source_root'])
    need(source_closure(paths['source_root']) == sources, 'Source changed during backend load')
    overlay_closure(config)
    # Serving source is a plain archived file: only6 naming literals and2
    # absolute import paths differ from the original, no runtime source rewriting.
    import uvicorn
    uvicorn.run(module.create_app(backend), host=args.host, port=args.port, workers=1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='brockone', description=__doc__)
    parser.add_argument('command', choices=('check-config', 'serve'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(argv)
    config, paths, sources = validate_config(args.config, args.config_sha256)
    if args.command == 'check-config':
        print(json.dumps({'state': 'configuration_verified_target_execution_pending', 'model': 'brockone', 'candidate': config['candidate'], 'source_files': len(sources)}))
        return 0
    serve(config, paths, sources)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
