#!/usr/bin/env python3
"""Re-extract the six evidence frames from the hash-verified original recording."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
    if sha256(args.source) != manifest['source']['sha256']:
        parser.error('Source SHA-256 differs from the original recording.')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Output directory must be empty; existing files will not be replaced.')
    frames = manifest['frames']
    selection = '+'.join(f'eq(n\\,{f["source_frame_index_zero_based"]})' for f in frames)
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=args.output) as temporary:
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '2',
            '-i', str(args.source.resolve()), '-map', '0:v:0',
            '-vf', 'select=' + selection, '-fps_mode', 'vfr',
            str(Path(temporary) / 'frame-%02d.png'),
        ], check=True)
        outputs = sorted(Path(temporary).glob('frame-*.png'))
        if len(outputs) != len(frames):
            raise RuntimeError('Decoded frame count did not match manifest.')
        for output, frame in zip(outputs, frames):
            output.rename(args.output / frame['file'])
    print(f'Extracted {len(frames)} original-resolution frames into {args.output}')


if __name__ == '__main__':
    main()
