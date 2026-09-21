# Native Python dependency repair

The serving venv uses `--system-site-packages` so it can import the TensorRT Python bindings installed with JetPack. The first native installation reached the final dependency check, which found two inherited-package problems: Ubuntu's SciPy 1.11.4 requires NumPy below 1.28, while the pinned Edge-LLM server requires NumPy 2.2.6; inherited PyNaCl 1.5.0 requires CFFI, whose package metadata was absent. This is dependency-installation evidence, not inference evidence.

Install compatible binary wheels into the existing venv:

```bash
python -m pip install --only-binary=:all: \
  'numpy==2.2.6' 'scipy==1.15.3' 'cffi==2.0.0'
python -m pip check
```

Run this with the venv's Python as the project owner, without `sudo`, `--user`, or `--break-system-packages`. Pip installs the selected SciPy in the venv ahead of the inherited copy. It must leave the distribution's Python packages in place. Keeping the explicit NumPy pin also prevents the compatibility repair from selecting a different NumPy version before subsequent server installation.

Verified public PyPI release metadata provides these CPython 3.12 Linux ARM64 wheels:

| Package | Python / dependency compatibility | Wheel | SHA-256 |
| --- | --- | --- | --- |
| SciPy 1.15.3 | Python >=3.10; NumPy >=1.23.5,<2.5 | `scipy-1.15.3-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `c05045d8b9bfd807ee1b9f38761993297b10b245f012b11b13b91ba8945f7e45` |
| CFFI 2.0.0 | Python >=3.9; pycparser on CPython | `cffi-2.0.0-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `b21e08af67b8a103c71a250401c78d5e0893beff75e28c53c98f4de42f774062` |

Wheel download sizes are 35,210,199 and 220,097 bytes respectively. The wheel-only option prevents an unexpected source build on the Nano. Source metadata: [SciPy 1.15.3](https://pypi.org/pypi/scipy/1.15.3/json), [CFFI 2.0.0](https://pypi.org/pypi/cffi/2.0.0/json). These are metadata hashes; this audit did not independently download or execute the ARM64 wheels.

The task's native provisioning script now records `pip check`, exact installed versions and module paths. Its CPU checks solve a tiny linear system through SciPy/NumPy, verify a deterministic test signature through inherited PyNaCl/CFFI, and require NumPy/SciPy/CFFI to resolve inside the venv. TensorRT import is recorded but no GPU operation is executed. Actual successful execution is recorded separately under `results/python-setup` on the Jetson; this audit alone does not establish that those checks passed.
