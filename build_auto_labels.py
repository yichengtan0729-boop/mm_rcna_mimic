from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from mm_rcna.config import load_config
from mm_rcna.data.auto_labels import AutoLabelBuilder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-csv", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    builder = AutoLabelBuilder(cfg)
    out = builder.build(args.output_csv)
    print(f"Saved to {out}")
    print(f"Updated studies CSV: {cfg.data.studies_csv}")


if __name__ == "__main__":
    main()