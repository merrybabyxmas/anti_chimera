from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.config import load_config
from anti_chimera.trainer_planner import train_planner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    train_planner(config, resume_checkpoint=args.resume)


if __name__ == '__main__':
    main()
