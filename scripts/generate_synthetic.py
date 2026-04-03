#!/usr/bin/env python3
"""
Usage:
  python scripts/generate_synthetic.py --n 100 --out data/synthetic/ --seed 0
  python scripts/generate_synthetic.py --n 50 --ref-dir data/raw/slf_cutouts/ # use real images as references
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentation.synthetic_generator import SyntheticConfig, SyntheticTrapGenerator


def main():
    parser = argparse.ArgumentParser(description="SLF Synthetic Dataset Generator")
    parser.add_argument(
        "--n", type=int, default=100, help="Number of images to generate"
    )
    parser.add_argument(
        "--out", type=str, default="data/synthetic/", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ref-dir",
        type=str,
        default=None,
        help="Directory of SLF reference PNGs with alpha",
    )
    parser.add_argument("--insects-min", type=int, default=1)
    parser.add_argument("--insects-max", type=int, default=8)
    args = parser.parse_args()

    config = SyntheticConfig(
        insects_per_image_range=(args.insects_min, args.insects_max),
        seed=args.seed,
    )
    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    gen = SyntheticTrapGenerator(config=config, slf_reference_dir=ref_dir)

    print(f"Generating {args.n} synthetic trap images...")
    images, annotations = gen.generate_batch(args.n)
    gen.save_batch(images, annotations, Path(args.out))
    print(f"Done. Images saved to {args.out}")


if __name__ == "__main__":
    main()
