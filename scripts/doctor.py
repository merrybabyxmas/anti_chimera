from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    print('=== anti_chimera doctor ===')
    print(f'python: {sys.version}')
    print(f'executable: {sys.executable}')
    print(f'cwd: {Path.cwd()}')
    print(f'PYTHONPATH: {os.environ.get("PYTHONPATH", "<unset>")}')

    try:
        import torch
        print(f'torch: {torch.__version__}')
        print(f'torch cuda runtime: {torch.version.cuda}')
        print(f'cuda available: {torch.cuda.is_available()}')
        print(f'cuda device count: {torch.cuda.device_count()}')
    except Exception as exc:
        print(f'torch import failed: {exc}')

    try:
        import diffusers
        print(f'diffusers: {diffusers.__version__}')
    except Exception as exc:
        print(f'diffusers import failed: {exc}')

    try:
        import transformers
        print(f'transformers: {transformers.__version__}')
    except Exception as exc:
        print(f'transformers import failed: {exc}')

    try:
        import anti_chimera
        print('anti_chimera import: ok')
    except Exception as exc:
        print(f'anti_chimera import failed: {exc}')

    hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
    print(f'hf cache exists: {hf_cache.exists()} -> {hf_cache}')
    if hf_cache.exists():
        for p in sorted(hf_cache.glob('models--*CogVideo*')):
            print(f'  found cache entry: {p}')


if __name__ == '__main__':
    main()
