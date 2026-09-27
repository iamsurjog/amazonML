"""Final test inference and submission validation."""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.blocking import DEFAULT_CANDIDATE_PATH, generate_candidate_pairs
from src.modeling import (
    DEFAULT_MODEL_PATH,
    MODEL_FEATURE_COLUMNS,
    prepare_model_features,
)
from src.pair_features import DEFAULT_PAIR_FEATURES_PATH, generate_pair_features
from src.preprocessing import DEFAULT_TSV_ROOT


PROJECT_ROOT = DEFAULT_TSV_ROOT.parents[1]
DEFAULT_MATCHING_RESULTS_PATH = PROJECT_ROOT / "matching_results.tsv"
SUBMISSION_COLUMNS = ("source1_entity_id", "matched_entity_ids")


def _validate_entity_id(value: object, *, description: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{description} is empty")
    entity_id = str(value)
    if not entity_id.strip() or any(
        character in entity_id for character in (",", "\t", "\r", "\n")
    ):
        raise ValueError(f"{description} is empty or contains an output delimiter: {entity_id!r}")
    return entity_id


def _source_paths(data_root: str | Path | None) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else DEFAULT_TSV_ROOT
    test_directory = root / "test"
    return {
        source: test_directory / f"test_{source}.tsv"
        for source in ("source1", "source2", "source3")
    }


def _iter_entity_id_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id"],
        dtype={"entity_id": "string"},
        encoding="utf-8-sig",
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
    )


def _build_entity_id_index(
    connection: sqlite3.Connection,
    source_paths: dict[str, Path],
    *,
    chunksize: int,
) -> int:
    connection.executescript(
        "CREATE TABLE source1_records ("
        "entity_id TEXT PRIMARY KEY, row_order INTEGER NOT NULL UNIQUE) WITHOUT ROWID;"
        "CREATE TABLE valid_target_ids ("
        "entity_id TEXT PRIMARY KEY, source_name TEXT NOT NULL) WITHOUT ROWID;"
        "CREATE TABLE predicted_links ("
        "source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, "
        "probability REAL NOT NULL, "
        "PRIMARY KEY (source1_entity_id, candidate_entity_id)) WITHOUT ROWID;"
    )

    source1_count = 0
    source1_buffer: list[tuple[str, int]] = []
    for frame in _iter_entity_id_chunks(source_paths["source1"], chunksize):
        for value in frame["entity_id"].tolist():
            source1_buffer.append(
                (_validate_entity_id(value, description="test Source 1 entity_id"), source1_count)
            )
            source1_count += 1
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO source1_records (entity_id, row_order) VALUES (?, ?)",
                    source1_buffer,
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("Test Source 1 entity_id values must be unique") from error
        source1_buffer.clear()

    target_buffer: list[tuple[str, str]] = []
    for source_name in ("source2", "source3"):
        for frame in _iter_entity_id_chunks(source_paths[source_name], chunksize):
            target_buffer.extend(
                (
                    _validate_entity_id(value, description=f"test {source_name} entity_id"),
                    source_name,
                )
                for value in frame["entity_id"].tolist()
            )
            try:
                with connection:
                    connection.executemany(
                        "INSERT INTO valid_target_ids (entity_id, source_name) VALUES (?, ?)",
                        target_buffer,
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "Source 2 and Source 3 entity_id values must be unique"
                ) from error
            target_buffer.clear()
    return source1_count


def validate_submission_frame(
    submission: pd.DataFrame,
    *,
    expected_source1_ids: Collection[str] | None = None,
    valid_target_ids: Collection[str] | None = None,
    id_index: sqlite3.Connection | None = None,
    validation_batch_size: int = 50_000,
) -> None:
    """Validate submission schema, coverage, list uniqueness, and target membership.

    Pass in-memory ID collections for small checks, or ``id_index`` from final
    inference to validate a large submission without loading millions of target
    IDs into a Python set. The submission DataFrame itself is always validated
    before the final output file is written.
    """
    if list(submission.columns) != list(SUBMISSION_COLUMNS):
        raise ValueError(
            f"Submission columns must be exactly {list(SUBMISSION_COLUMNS)}; "
            f"found {list(submission.columns)}"
        )
    if validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")

    source1_ids = submission["source1_entity_id"].astype("string")
    for value in source1_ids:
        _validate_entity_id(value, description="submission source1_entity_id")
    if bool(source1_ids.duplicated().any()):
        raise ValueError("Submission contains duplicate Source 1 IDs")
    if expected_source1_ids is not None:
        expected = set(expected_source1_ids)
        if len(source1_ids) != len(expected) or set(source1_ids.tolist()) != expected:
            raise ValueError("Submission must contain exactly one row for every test Source 1 ID")

    if id_index is not None:
        source1_count = int(
            id_index.execute("SELECT COUNT(*) FROM source1_records").fetchone()[0]
        )
        if len(source1_ids) != source1_count:
            raise ValueError(
                f"Submission row count {len(source1_ids)} does not match "
                f"test Source 1 count {source1_count}"
            )
        id_index.execute(
            "CREATE TEMP TABLE IF NOT EXISTS submission_source1_check ("
            "entity_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        id_index.execute(
            "CREATE TEMP TABLE IF NOT EXISTS submission_target_check ("
            "entity_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        with id_index:
            id_index.execute("DELETE FROM submission_source1_check")
            id_index.executemany(
                "INSERT INTO submission_source1_check (entity_id) VALUES (?)",
                ((str(entity_id),) for entity_id in source1_ids),
            )
        missing_source1 = id_index.execute(
            "SELECT s.entity_id FROM submission_source1_check AS s "
            "LEFT JOIN source1_records AS r USING (entity_id) "
            "WHERE r.entity_id IS NULL LIMIT 1"
        ).fetchone()
        if missing_source1 is not None:
            raise ValueError(f"Unknown test Source 1 ID in submission: {missing_source1[0]!r}")
    elif expected_source1_ids is None:
        raise ValueError("expected_source1_ids or id_index is required for coverage validation")

    target_buffer: list[tuple[str]] = []
    for source1_id, matched_entity_ids in submission.itertuples(index=False, name=None):
        if not isinstance(matched_entity_ids, str):
            raise ValueError(f"Match list for {source1_id!r} must be a string")
        match_ids = matched_entity_ids.split(",") if matched_entity_ids else []
        if any(not match_id or match_id != match_id.strip() for match_id in match_ids):
            raise ValueError(f"Match list for {source1_id!r} contains an empty or malformed ID")
        if len(match_ids) != len(set(match_ids)):
            raise ValueError(f"Match list for {source1_id!r} contains duplicate IDs")
        for match_id in match_ids:
            _validate_entity_id(match_id, description=f"match ID for {source1_id}")
            if valid_target_ids is not None and match_id not in valid_target_ids:
                raise ValueError(f"Match ID {match_id!r} is not from Source 2 or Source 3")
            if id_index is not None:
                target_buffer.append((match_id,))
                if len(target_buffer) >= validation_batch_size:
                    _insert_submission_target_batch(id_index, target_buffer)
                    target_buffer.clear()
    if id_index is not None:
        if target_buffer:
            _insert_submission_target_batch(id_index, target_buffer)
        missing_target = id_index.execute(
            "SELECT s.entity_id FROM submission_target_check AS s "
            "LEFT JOIN valid_target_ids AS t USING (entity_id) "
            "WHERE t.entity_id IS NULL LIMIT 1"
        ).fetchone()
        if missing_target is not None:
            raise ValueError(
                f"Match ID {missing_target[0]!r} is not from Source 2 or Source 3"
            )
    elif valid_target_ids is None:
        raise ValueError("valid_target_ids or id_index is required for match-ID validation")


def _insert_submission_target_batch(
    connection: sqlite3.Connection,
    target_ids: list[tuple[str]],
) -> None:
    with connection:
        connection.executemany(
            "INSERT OR IGNORE INTO submission_target_check (entity_id) VALUES (?)",
            target_ids,
        )


def _load_model(model_path: Path) -> tuple[Any, float]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained matcher model does not exist: {model_path}")
    try:
        import joblib

        bundle = joblib.load(model_path)
    except Exception as error:
        raise RuntimeError(f"Could not load matcher model: {model_path}") from error
    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        raise ValueError("Model artifact must contain 'model' and 'threshold'")
    if tuple(bundle.get("feature_columns", ())) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Model feature columns do not match the current feature pipeline")
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Model threshold is invalid: {threshold}")
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise TypeError("Saved matcher does not implement predict_proba")
    classes = list(getattr(model, "classes_", ()))
    if 1 not in classes:
        raise ValueError("Saved matcher has no positive class (1)")
    return model, threshold


def _feature_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep="\t",
        dtype={
            "source1_entity_id": "string",
            "candidate_entity_id": "string",
            "candidate_source": "string",
        },
        chunksize=chunksize,
    )


def _store_passing_predictions(
    connection: sqlite3.Connection,
    model: Any,
    threshold: float,
    feature_path: Path,
    *,
    chunksize: int,
) -> int:
    positive_class_index = list(model.classes_).index(1)
    passed_count = 0
    for frame in _feature_chunks(feature_path, chunksize):
        features = prepare_model_features(frame)
        probabilities = model.predict_proba(features)[:, positive_class_index]
        if not np.isfinite(probabilities).all():
            raise ValueError("Matcher produced non-finite probabilities")
        passing = np.flatnonzero(probabilities >= threshold)
        source1_ids = frame["source1_entity_id"].tolist()
        candidate_ids = frame["candidate_entity_id"].tolist()
        rows = [
            (
                _validate_entity_id(
                    source1_ids[row_index],
                    description="pair-feature Source 1 ID",
                ),
                _validate_entity_id(
                    candidate_ids[row_index],
                    description="pair-feature candidate ID",
                ),
                float(probabilities[row_index]),
            )
            for row_index in passing
        ]
        if rows:
            with connection:
                connection.executemany(
                    "INSERT INTO predicted_links "
                    "(source1_entity_id, candidate_entity_id, probability) VALUES (?, ?, ?) "
                    "ON CONFLICT (source1_entity_id, candidate_entity_id) "
                    "DO UPDATE SET probability = MAX(probability, excluded.probability)",
                    rows,
                )
            passed_count += len(rows)
    return passed_count


def _submission_frame(
    connection: sqlite3.Connection,
    *,
    max_matches_per_source1: int | None,
) -> pd.DataFrame:
    if max_matches_per_source1 is not None and max_matches_per_source1 <= 0:
        raise ValueError("max_matches_per_source1 must be positive or None")
    output_rows: list[tuple[str, str]] = []
    cursor = connection.execute(
        "SELECT s.entity_id, p.candidate_entity_id "
        "FROM source1_records AS s "
        "LEFT JOIN predicted_links AS p ON p.source1_entity_id = s.entity_id "
        "ORDER BY s.row_order, p.probability DESC, p.candidate_entity_id ASC"
    )
    current_source1_id: str | None = None
    current_matches: list[str] = []
    for source1_id, candidate_id in cursor:
        if source1_id != current_source1_id:
            if current_source1_id is not None:
                output_rows.append((current_source1_id, ",".join(current_matches)))
            current_source1_id = source1_id
            current_matches = []
        if candidate_id is not None and (
            max_matches_per_source1 is None
            or len(current_matches) < max_matches_per_source1
        ):
            current_matches.append(candidate_id)
    if current_source1_id is not None:
        output_rows.append((current_source1_id, ",".join(current_matches)))
    return pd.DataFrame(output_rows, columns=SUBMISSION_COLUMNS, dtype="string")


def generate_matching_results(
    *,
    data_root: str | Path | None = None,
    candidate_path: str | Path | None = None,
    features_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path = DEFAULT_MATCHING_RESULTS_PATH,
    idf_path: str | Path | None = None,
    working_directory: str | Path | None = None,
    chunksize: int = 50_000,
    candidate_chunksize: int = 10_000,
    pair_batch_size: int = 10_000,
    max_candidates_per_source1: int | None = 500,
    max_matches_per_source1: int | None = None,
    regenerate_candidates: bool = False,
    regenerate_features: bool = False,
) -> Path:
    """Generate, validate, then atomically save test ``matching_results.tsv``."""
    if chunksize <= 0 or candidate_chunksize <= 0 or pair_batch_size <= 0:
        raise ValueError("chunk sizes must be positive integers")

    matcher_path = Path(model_path)
    model, threshold = _load_model(matcher_path)
    source_paths = _source_paths(data_root)
    missing_sources = [path for path in source_paths.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "Missing test source TSV file(s): "
            + ", ".join(str(path) for path in missing_sources)
        )

    candidates = Path(candidate_path) if candidate_path is not None else DEFAULT_CANDIDATE_PATH
    if regenerate_candidates or not candidates.is_file():
        generate_candidate_pairs(
            "test",
            data_root=data_root,
            output_path=candidates,
            working_directory=working_directory,
            chunksize=candidate_chunksize,
            max_candidates_per_source1=max_candidates_per_source1,
        )

    features = (
        Path(features_path)
        if features_path is not None
        else DEFAULT_PAIR_FEATURES_PATH
    )
    source_mtime = max(path.stat().st_mtime_ns for path in source_paths.values())
    features_are_stale = (
        not features.is_file()
        or candidates.stat().st_mtime_ns > features.stat().st_mtime_ns
        or source_mtime > features.stat().st_mtime_ns
    )
    if regenerate_features or features_are_stale:
        generate_pair_features(
            "test",
            candidate_path=candidates,
            data_root=data_root,
            output_path=features,
            idf_path=idf_path,
            working_directory=working_directory,
            chunksize=chunksize,
            pair_batch_size=pair_batch_size,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() in {
        path.resolve() for path in source_paths.values()
    } | {candidates.resolve(), features.resolve(), matcher_path.resolve()}:
        raise ValueError("output_path must not overwrite an inference input")
    if idf_path is not None and destination.resolve() == Path(idf_path).resolve():
        raise ValueError("output_path must not overwrite TF-IDF statistics")
    work_dir = (
        Path(working_directory) if working_directory is not None else destination.parent
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    temporary_output: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="entity-matching-results-", dir=work_dir) as temp_dir:
            connection = sqlite3.connect(Path(temp_dir) / "submission.sqlite")
            try:
                connection.execute("PRAGMA journal_mode = OFF")
                connection.execute("PRAGMA synchronous = OFF")
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute("PRAGMA cache_size = -65536")
                _build_entity_id_index(
                    connection,
                    source_paths,
                    chunksize=chunksize,
                )
                _store_passing_predictions(
                    connection,
                    model,
                    threshold,
                    features,
                    chunksize=chunksize,
                )
                connection.execute(
                    "CREATE INDEX predicted_links_by_source1_score "
                    "ON predicted_links (source1_entity_id, probability DESC, candidate_entity_id)"
                )
                submission = _submission_frame(
                    connection,
                    max_matches_per_source1=max_matches_per_source1,
                )
                validate_submission_frame(submission, id_index=connection)

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=destination.parent,
                    delete=False,
                ) as output_stream:
                    temporary_output = Path(output_stream.name)
                    submission.to_csv(output_stream, sep="\t", index=False)
            finally:
                connection.close()
        os.replace(temporary_output, destination)
        temporary_output = None
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local test inference and validate matching_results.tsv."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSV_ROOT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_PAIR_FEATURES_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MATCHING_RESULTS_PATH)
    parser.add_argument("--idf", type=Path, default=None)
    parser.add_argument("--working-directory", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--candidate-chunk-size", type=int, default=10_000)
    parser.add_argument("--pair-batch-size", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--regenerate-candidates", action="store_true")
    parser.add_argument("--regenerate-features", action="store_true")
    args = parser.parse_args()
    output = generate_matching_results(
        data_root=args.data_root,
        candidate_path=args.candidates,
        features_path=args.features,
        model_path=args.model,
        output_path=args.output,
        idf_path=args.idf,
        working_directory=args.working_directory,
        chunksize=args.chunk_size,
        candidate_chunksize=args.candidate_chunk_size,
        pair_batch_size=args.pair_batch_size,
        max_candidates_per_source1=args.max_candidates or None,
        max_matches_per_source1=args.max_matches or None,
        regenerate_candidates=args.regenerate_candidates,
        regenerate_features=args.regenerate_features,
    )
    print(f"Validated and wrote matching results to {output}")


if __name__ == "__main__":
    main()
