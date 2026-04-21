from __future__ import annotations

import argparse

from anti_chimera.config import load_config
from anti_chimera.trainer import train
from anti_chimera.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config['seed']))
    train(config)


if __name__ == '__main__':
    main()
