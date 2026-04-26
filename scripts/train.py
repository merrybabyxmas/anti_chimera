from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.config import load_config
from anti_chimera.trainer import train as train_lite
from anti_chimera.trainer_cogvideox import train as train_cogvideox_legacy
from anti_chimera.trainer_cogvideox_v2 import train as train_cogvideox_v2
from anti_chimera.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Optional checkpoint to resume from.')
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config['seed']))
    backend = str(config.get('model', {}).get('backend', 'lite3d'))
    trainer_variant = str(config.get('training', {}).get('cogvideox_trainer', 'v2')).lower()
    if backend == 'cogvideox':
        if trainer_variant == 'legacy':
            train_cogvideox_legacy(config, resume_checkpoint=args.resume)
        else:
            train_cogvideox_v2(config, resume_checkpoint=args.resume)
    else:
        train_lite(config, resume_checkpoint=args.resume)


if __name__ == '__main__':
    main()
