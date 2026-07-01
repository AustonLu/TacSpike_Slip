from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a derived score cache by averaging selected rows.")
    parser.add_argument("--score-cache", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--rows", default=None, help="Comma-separated row ids. Omit to use all rows.")
    parser.add_argument("--mode", choices=("copy", "mean"), default="mean")
    args = parser.parse_args()

    cache = np.load(args.score_cache, allow_pickle=False)
    matrix = cache["raw_score_matrix"].astype(np.float64, copy=False)
    if args.rows:
        rows = [int(part) for part in args.rows.split(",") if part.strip()]
        selected = matrix[rows]
    else:
        rows = list(range(matrix.shape[0]))
        selected = matrix
    if args.mode == "mean":
        output_matrix = selected.mean(axis=0, keepdims=True).astype(np.float32)
    else:
        output_matrix = selected.astype(np.float32)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        raw_score_matrix=output_matrix,
        labels=cache["labels"],
        global_indices=cache["global_indices"],
        seq_offsets=cache["seq_offsets"],
        selected_sequences=cache["selected_sequences"],
        model_records_json=np.asarray(
            json.dumps({"source": str(args.score_cache), "rows": rows, "mode": args.mode}, sort_keys=True, default=json_default)
        ),
    )
    print(json.dumps({"output": str(args.output_npz), "rows": rows, "mode": args.mode, "shape": list(output_matrix.shape)}))


if __name__ == "__main__":
    main()
