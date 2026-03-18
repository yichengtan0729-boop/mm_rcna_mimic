from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from mm_rcna.data.auto_labels import AutoLabelBuilder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input-csv', required=True)
    p.add_argument('--output-csv', required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input_csv)
    builder = AutoLabelBuilder()
    out = builder.build(df)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
