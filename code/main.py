"""Command-line entry point for local entity-resolution blocking."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.blocking import generate_candidate_pairs
from src.preprocessing import DEFAULT_TSV_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Source 1-to-Source 2/3 blocking candidates from local TSVs."
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_TSV_ROOT,
        help="Directory containing train/ and test/ TSV folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path (defaults to candidate_pairs.tsv for test or "
            "candidate_pairs_train.tsv for train)"
        ),
    )
    parser.add_argument(
        "--working-directory",
        type=Path,
        default=None,
        help="Disk location for the temporary SQLite LSH index",
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=500,
        help="Maximum candidates per Source 1 row; use 0 for no cap",
    )
    args = parser.parse_args()

    output_path = generate_candidate_pairs(
        args.split,
        data_root=args.data_root,
        output_path=args.output,
        working_directory=args.working_directory,
        chunksize=args.chunk_size,
        max_candidates_per_source1=args.max_candidates or None,
    )
    print(f"Wrote candidate pairs to {output_path}")


if __name__ == "__main__":
    main()
