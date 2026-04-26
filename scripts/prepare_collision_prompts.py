from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.data.collision_planner import default_collision_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a text-only two-entity collision prompt set.")
    parser.add_argument("--output", type=Path, default=Path("data/collision_prompts/collision_prompts_20.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = default_collision_prompts()
    with args.output.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
