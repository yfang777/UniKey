"""Generate seeded cube-color scenes. Run from the repository root."""

import argparse
from pathlib import Path

from .generator import SceneGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num < 1:
        parser.error("--num must be positive")
    generator = SceneGenerator(
        "arrange_cube_by_color",
        "configurations/initializations_range/arrange_cube_by_color/arrange_cube_by_color.yaml",
        "assets/objects/arrange_cube_by_color",
    )
    generator.generate_samples(args.base_dir, args.num, args.seed)
    print(f"Generated {args.num} scenes in {args.base_dir}")


if __name__ == "__main__":
    main()
