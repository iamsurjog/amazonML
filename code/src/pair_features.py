"""Streaming pairwise similarity features for blocked entity pairs."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.blocking import DEFAULT_CANDIDATE_PATH
from src.preprocessing import DEFAULT_TSV_ROOT, SourceSplit, iter_source_tsv


PROJECT_ROOT = DEFAULT_TSV_ROOT.parents[1]
DEFAULT_TFIDF_IDF_PATH = PROJECT_ROOT / "tfidf_idf_train.npz"
DEFAULT_PAIR_FEATURES_PATH = PROJECT_ROOT / "pair_features.tsv"
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")

PAIR_FEATURE_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "name_exact_match",
    "address_exact_match",
    "name_jaccard_similarity",
    "address_jaccard_similarity",
    "name_levenshtein_distance",
    "address_levenshtein_distance",
    "name_levenshtein_similarity",
    "address_levenshtein_similarity",
    "name_tfidf_cosine_similarity",
    "address_tfidf_cosine_similarity",
    "name_exact_token_match",
    "address_exact_token_match",
    "name_token_overlap_count",
    "address_token_overlap_count",
    "name_token_containment",
    "address_token_containment",
    "name_token_count_source1",
    "name_token_count_candidate",
    "address_token_count_source1",
    "address_token_count_candidate",
    "name_character_length_ratio",
    "address_character_length_ratio",
    "address_exact_numeric_token_match",
    "address_exact_postal_code_match",
    "address_first_number_match",
    "country_exact_match",
)

_NUMERIC_TOKEN_RE = re.compile(r"(?<!\w)\d+(?!\w)")


def _country_key(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _checked_entity_id(value: object, *, source: str) -> str:
    if value is None:
        raise ValueError(f"{source} contains an empty entity_id")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        raise ValueError(f"{source} contains an empty entity_id")
    entity_id = str(value)
    if not entity_id.strip():
        raise ValueError(f"{source} contains an empty entity_id")
    if any(character in entity_id for character in (",", "\t", "\r", "\n")):
        raise ValueError(f"{source} entity_id {entity_id!r} contains an output delimiter")
    return entity_id


def _source_paths(split: SourceSplit, data_root: str | Path | None) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else DEFAULT_TSV_ROOT
    split_directory = root / split
    return {
        source: split_directory / f"{split}_{source}.tsv"
        for source in ("source1", "source2", "source3")
    }


def _new_hashing_vectorizer(
    vectorizer_class: Any,
    *,
    n_features: int,
    ngram_range: tuple[int, int],
) -> Any:
    return vectorizer_class(
        analyzer="char_wb",
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        lowercase=False,
        norm=None,
        dtype=np.float32,
    )


def fit_tfidf_statistics(
    *,
    data_root: str | Path | None = None,
    split: SourceSplit = "train",
    output_path: str | Path | None = None,
    chunksize: int = 50_000,
    n_features: int = 1 << 20,
    ngram_range: tuple[int, int] = (2, 5),
) -> Path:
    """Fit and save streaming smoothed IDF arrays for normalized names and addresses.

    Hashing keeps a fixed feature space, while chunked document-frequency counts
    avoid building a corpus-sized sparse matrix. The resulting statistics can be
    reused for both training and test pair features.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    if chunksize <= 0 or n_features <= 0:
        raise ValueError("chunksize and n_features must be positive integers")
    if min(ngram_range) <= 0 or ngram_range[0] > ngram_range[1]:
        raise ValueError(f"Invalid ngram_range: {ngram_range}")

    try:
        from sklearn.feature_extraction.text import HashingVectorizer
    except ImportError as error:
        raise RuntimeError("TF-IDF features require scikit-learn") from error

    paths = _source_paths(split, data_root)
    missing_paths = [path for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing source TSV file(s): " + ", ".join(str(path) for path in missing_paths)
        )

    default_statistics_path = PROJECT_ROOT / f"tfidf_idf_{split}.npz"
    destination = Path(output_path) if output_path is not None else default_statistics_path
    if destination.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("TF-IDF statistics output must not overwrite an input source TSV")

    field_columns = {
        "name": "business_name_normalized",
        "address": "business_address_normalized",
    }
    vectorizers = {
        field: _new_hashing_vectorizer(
            HashingVectorizer,
            n_features=n_features,
            ngram_range=ngram_range,
        )
        for field in field_columns
    }
    document_frequencies = {
        field: np.zeros(n_features, dtype=np.int64) for field in field_columns
    }
    document_count = 0

    for path in paths.values():
        for frame in iter_source_tsv(path, chunksize=chunksize, normalize=True):
            document_count += len(frame)
            for field, column in field_columns.items():
                texts = frame[column].fillna("").astype("string").tolist()
                matrix = vectorizers[field].transform(texts)
                document_frequencies[field] += np.asarray(
                    matrix.getnnz(axis=0), dtype=np.int64
                ).reshape(-1)

    idf = {
        field: (
            np.log((1.0 + document_count) / (1.0 + frequencies)) + 1.0
        ).astype(np.float32)
        for field, frequencies in document_frequencies.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        np.savez_compressed(
            temporary_path,
            name_idf=idf["name"],
            address_idf=idf["address"],
            n_features=np.asarray(n_features, dtype=np.int64),
            ngram_range=np.asarray(ngram_range, dtype=np.int64),
            fit_split=np.asarray(split),
            format_version=np.asarray(1, dtype=np.int64),
        )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _load_tfidf_statistics(
    path: str | Path,
    *,
    n_features: int,
    ngram_range: tuple[int, int],
) -> dict[str, np.ndarray]:
    statistics_path = Path(path)
    if not statistics_path.is_file():
        raise FileNotFoundError(f"TF-IDF statistics do not exist: {statistics_path}")
    with np.load(statistics_path, allow_pickle=False) as stored:
        stored_n_features = int(stored["n_features"])
        stored_ngram_range = tuple(int(value) for value in stored["ngram_range"])
        if stored_n_features != n_features or stored_ngram_range != ngram_range:
            raise ValueError(
                "TF-IDF statistics use a different feature configuration: "
                f"n_features={stored_n_features}, ngram_range={stored_ngram_range}"
            )
        if int(stored["format_version"]) != 1:
            raise ValueError("Unsupported TF-IDF statistics format version")
        name_idf = stored["name_idf"].astype(np.float32, copy=True)
        address_idf = stored["address_idf"].astype(np.float32, copy=True)
    if len(name_idf) != n_features or len(address_idf) != n_features:
        raise ValueError("TF-IDF statistics have an invalid feature dimension")
    return {"name": name_idf, "address": address_idf}


def _token_metrics(left: str, right: str) -> tuple[int, int, int, float, float]:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    jaccard = len(overlap) / len(union) if union else 0.0
    smaller_count = min(len(left_tokens), len(right_tokens))
    containment = len(overlap) / smaller_count if smaller_count else 0.0
    return len(left_tokens), len(right_tokens), len(overlap), jaccard, containment


def _character_length_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    return min(len(left), len(right)) / longest if longest else 0.0


def _edit_similarity(left: str, right: str, distance: int) -> float:
    longest = max(len(left), len(right))
    return 1.0 - distance / longest if longest else 0.0


def compute_pairwise_string_features(
    source1_name: str,
    candidate_name: str,
    source1_address: str,
    candidate_address: str,
    *,
    name_tfidf_cosine: float = 0.0,
    address_tfidf_cosine: float = 0.0,
    source1_country: str = "",
    candidate_country: str = "",
) -> dict[str, int | float]:
    """Compute non-vector pair features from Phase 1 normalized text."""
    name_left = source1_name or ""
    name_right = candidate_name or ""
    address_left = source1_address or ""
    address_right = candidate_address or ""

    (
        name_left_count,
        name_right_count,
        name_overlap_count,
        name_jaccard,
        name_containment,
    ) = _token_metrics(name_left, name_right)
    (
        address_left_count,
        address_right_count,
        address_overlap_count,
        address_jaccard,
        address_containment,
    ) = _token_metrics(address_left, address_right)
    name_distance = int(Levenshtein.distance(name_left, name_right))
    address_distance = int(Levenshtein.distance(address_left, address_right))
    source1_address_numbers = _NUMERIC_TOKEN_RE.findall(address_left)
    candidate_address_numbers = _NUMERIC_TOKEN_RE.findall(address_right)
    shared_numbers = set(source1_address_numbers) & set(candidate_address_numbers)
    shared_postal_length_numbers = {
        number for number in shared_numbers if 3 <= len(number) <= 10
    }

    features: dict[str, int | float] = {
        "name_exact_match": int(bool(name_left) and name_left == name_right),
        "address_exact_match": int(bool(address_left) and address_left == address_right),
        "name_jaccard_similarity": name_jaccard,
        "address_jaccard_similarity": address_jaccard,
        "name_levenshtein_distance": name_distance,
        "address_levenshtein_distance": address_distance,
        "name_levenshtein_similarity": _edit_similarity(
            name_left, name_right, name_distance
        ),
        "address_levenshtein_similarity": _edit_similarity(
            address_left, address_right, address_distance
        ),
        "name_tfidf_cosine_similarity": float(name_tfidf_cosine),
        "address_tfidf_cosine_similarity": float(address_tfidf_cosine),
        "name_exact_token_match": int(name_overlap_count > 0),
        "address_exact_token_match": int(address_overlap_count > 0),
        "name_token_overlap_count": name_overlap_count,
        "address_token_overlap_count": address_overlap_count,
        "name_token_containment": name_containment,
        "address_token_containment": address_containment,
        "name_token_count_source1": name_left_count,
        "name_token_count_candidate": name_right_count,
        "address_token_count_source1": address_left_count,
        "address_token_count_candidate": address_right_count,
        "name_character_length_ratio": _character_length_ratio(name_left, name_right),
        "address_character_length_ratio": _character_length_ratio(
            address_left, address_right
        ),
        "address_exact_numeric_token_match": int(bool(shared_numbers)),
        "address_exact_postal_code_match": int(bool(shared_postal_length_numbers)),
        "address_first_number_match": int(
            bool(source1_address_numbers)
            and bool(candidate_address_numbers)
            and source1_address_numbers[0] == candidate_address_numbers[0]
        ),
        "country_exact_match": int(
            bool(_country_key(source1_country))
            and _country_key(source1_country) == _country_key(candidate_country)
        ),
    }
    return features


def _weighted_l2_normalize(matrix: Any, idf: np.ndarray, normalize: Any) -> Any:
    weighted = matrix.tocsr(copy=True)
    if weighted.nnz:
        weighted.data *= idf[weighted.indices]
        weighted = normalize(weighted, norm="l2", axis=1, copy=False)
    return weighted


def _paired_tfidf_cosine(
    left_texts: list[str],
    right_texts: list[str],
    *,
    vectorizer: Any,
    idf: np.ndarray,
    normalize: Any,
) -> np.ndarray:
    pair_count = len(left_texts)
    if pair_count == 0:
        return np.empty(0, dtype=np.float32)
    matrix = vectorizer.transform(left_texts + right_texts)
    matrix = _weighted_l2_normalize(matrix, idf, normalize)
    similarities = matrix[:pair_count].multiply(matrix[pair_count:]).sum(axis=1)
    return np.asarray(similarities, dtype=np.float32).reshape(-1)


def _format_feature_value(value: object) -> str | int:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    return str(value)


def _build_record_store(
    connection: sqlite3.Connection,
    *,
    source_paths: dict[str, Path],
    chunksize: int,
) -> int:
    connection.execute(
        "CREATE TABLE records ("
        "entity_id TEXT PRIMARY KEY, source_name TEXT NOT NULL, "
        "business_name TEXT NOT NULL, business_address TEXT NOT NULL, country TEXT NOT NULL"
        ") WITHOUT ROWID"
    )
    inserted_rows = 0
    for source_name, path in source_paths.items():
        for frame in iter_source_tsv(path, chunksize=chunksize, normalize=True):
            records = []
            for entity_id, name, address, country in zip(
                frame["entity_id"].tolist(),
                frame["business_name_normalized"].tolist(),
                frame["business_address_normalized"].tolist(),
                frame["country"].tolist(),
                strict=True,
            ):
                records.append(
                    (
                        _checked_entity_id(entity_id, source=source_name),
                        source_name,
                        str(name),
                        str(address),
                        _country_key(country),
                    )
                )
            try:
                with connection:
                    connection.executemany(
                        "INSERT INTO records "
                        "(entity_id, source_name, business_name, business_address, country) "
                        "VALUES (?, ?, ?, ?, ?)",
                        records,
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "entity_id values must be unique across all three sources"
                ) from error
            inserted_rows += len(records)
    return inserted_rows


def _write_pair_batch(
    connection: sqlite3.Connection,
    pair_batch: list[tuple[str, str]],
    *,
    writer: Any,
    vectorizers: dict[str, Any],
    idf: dict[str, np.ndarray],
    normalize: Any,
) -> int:
    if not pair_batch:
        return 0
    connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS candidate_batch ("
        "pair_row INTEGER PRIMARY KEY, source1_entity_id TEXT NOT NULL, "
        "candidate_entity_id TEXT NOT NULL)"
    )
    with connection:
        connection.execute("DELETE FROM candidate_batch")
        connection.executemany(
            "INSERT INTO candidate_batch "
            "(pair_row, source1_entity_id, candidate_entity_id) VALUES (?, ?, ?)",
            (
                (row_index, source1_id, candidate_id)
                for row_index, (source1_id, candidate_id) in enumerate(pair_batch)
            ),
        )

    rows = connection.execute(
        "SELECT p.pair_row, p.source1_entity_id, p.candidate_entity_id, "
        "s1.source_name, s1.business_name, s1.business_address, s1.country, "
        "c.source_name, c.business_name, c.business_address, c.country "
        "FROM candidate_batch AS p "
        "LEFT JOIN records AS s1 ON s1.entity_id = p.source1_entity_id "
        "LEFT JOIN records AS c ON c.entity_id = p.candidate_entity_id "
        "ORDER BY p.pair_row"
    ).fetchall()
    if len(rows) != len(pair_batch):
        raise RuntimeError("Feature join returned an unexpected number of candidate pairs")

    for row in rows:
        if row[3] != "source1":
            raise ValueError(f"Candidate references unknown or non-Source-1 ID {row[1]!r}")
        if row[7] not in ("source2", "source3"):
            raise ValueError(
                f"Candidate ID {row[2]!r} is missing or does not belong to Source 2/3"
            )

    source1_names = [row[4] or "" for row in rows]
    candidate_names = [row[8] or "" for row in rows]
    source1_addresses = [row[5] or "" for row in rows]
    candidate_addresses = [row[9] or "" for row in rows]
    name_cosines = _paired_tfidf_cosine(
        source1_names,
        candidate_names,
        vectorizer=vectorizers["name"],
        idf=idf["name"],
        normalize=normalize,
    )
    address_cosines = _paired_tfidf_cosine(
        source1_addresses,
        candidate_addresses,
        vectorizer=vectorizers["address"],
        idf=idf["address"],
        normalize=normalize,
    )

    for row, name_cosine, address_cosine in zip(
        rows, name_cosines, address_cosines, strict=True
    ):
        feature_values = compute_pairwise_string_features(
            row[4] or "",
            row[8] or "",
            row[5] or "",
            row[9] or "",
            name_tfidf_cosine=float(name_cosine),
            address_tfidf_cosine=float(address_cosine),
            source1_country=row[6] or "",
            candidate_country=row[10] or "",
        )
        output_values = (
            row[1],
            row[2],
            row[7],
            *(feature_values[column] for column in PAIR_FEATURE_COLUMNS[3:]),
        )
        writer.writerow(tuple(_format_feature_value(value) for value in output_values))
    return len(rows)


def generate_pair_features(
    split: SourceSplit = "test",
    *,
    candidate_path: str | Path | None = None,
    data_root: str | Path | None = None,
    output_path: str | Path | None = None,
    idf_path: str | Path | None = None,
    fit_idf_split: SourceSplit = "train",
    working_directory: str | Path | None = None,
    chunksize: int = 50_000,
    pair_batch_size: int = 10_000,
    n_features: int = 1 << 20,
    ngram_range: tuple[int, int] = (2, 5),
) -> Path:
    """Compute pair features for every candidate listed in a Phase 2 TSV.

    Records and candidate pairs are processed in chunks. A temporary SQLite
    record store provides ID lookups without loading the full sources into RAM.
    TF-IDF cosine features use the saved training IDF statistics by default.
    """
    if split not in ("train", "test") or fit_idf_split not in ("train", "test"):
        raise ValueError("split and fit_idf_split must be 'train' or 'test'")
    if chunksize <= 0 or pair_batch_size <= 0:
        raise ValueError("chunksize and pair_batch_size must be positive integers")
    if n_features <= 0 or min(ngram_range) <= 0 or ngram_range[0] > ngram_range[1]:
        raise ValueError("Invalid TF-IDF feature configuration")

    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize
    except ImportError as error:
        raise RuntimeError("Pair TF-IDF features require scikit-learn") from error

    paths = _source_paths(split, data_root)
    missing_paths = [path for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing source TSV file(s): " + ", ".join(str(path) for path in missing_paths)
        )

    default_candidate_path = (
        DEFAULT_CANDIDATE_PATH
        if split == "test"
        else DEFAULT_CANDIDATE_PATH.with_name("candidate_pairs_train.tsv")
    )
    candidates_path = (
        Path(candidate_path) if candidate_path is not None else default_candidate_path
    )
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Candidate TSV does not exist: {candidates_path}")

    destination = (
        Path(output_path)
        if output_path is not None
        else (
            DEFAULT_PAIR_FEATURES_PATH
            if split == "test"
            else PROJECT_ROOT / "pair_features_train.tsv"
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() in {path.resolve() for path in paths.values()} | {
        candidates_path.resolve()
    }:
        raise ValueError("output_path must not overwrite an input TSV")

    default_statistics_path = PROJECT_ROOT / f"tfidf_idf_{fit_idf_split}.npz"
    statistics_path = Path(idf_path) if idf_path is not None else default_statistics_path
    if destination.resolve() == statistics_path.resolve():
        raise ValueError("Pair-feature output must not overwrite TF-IDF statistics")
    if not statistics_path.is_file():
        fit_tfidf_statistics(
            data_root=data_root,
            split=fit_idf_split,
            output_path=statistics_path,
            chunksize=chunksize,
            n_features=n_features,
            ngram_range=ngram_range,
        )
    idf = _load_tfidf_statistics(
        statistics_path,
        n_features=n_features,
        ngram_range=ngram_range,
    )
    vectorizers = {
        field: _new_hashing_vectorizer(
            HashingVectorizer,
            n_features=n_features,
            ngram_range=ngram_range,
        )
        for field in ("name", "address")
    }

    work_dir = (
        Path(working_directory) if working_directory is not None else destination.parent
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="entity-pair-features-", dir=work_dir) as temp_dir:
            database_path = Path(temp_dir) / "records.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("PRAGMA journal_mode = OFF")
                connection.execute("PRAGMA synchronous = OFF")
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute("PRAGMA cache_size = -32768")
                _build_record_store(
                    connection,
                    source_paths=paths,
                    chunksize=chunksize,
                )

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
                    writer = csv.writer(
                        output_stream,
                        delimiter="\t",
                        lineterminator="\n",
                    )
                    writer.writerow(PAIR_FEATURE_COLUMNS)
                    with candidates_path.open(
                        "r", encoding="utf-8-sig", newline=""
                    ) as candidates_stream:
                        candidate_reader = csv.DictReader(
                            candidates_stream,
                            delimiter="\t",
                        )
                        if candidate_reader.fieldnames is None:
                            raise ValueError("Candidate TSV is empty or has no header")
                        missing_candidate_columns = [
                            column
                            for column in CANDIDATE_COLUMNS
                            if column not in candidate_reader.fieldnames
                        ]
                        if missing_candidate_columns:
                            raise ValueError(
                                "Candidate TSV is missing required columns "
                                f"{missing_candidate_columns}"
                            )

                        pair_batch: list[tuple[str, str]] = []
                        for line_number, candidate_row in enumerate(
                            candidate_reader, start=2
                        ):
                            source1_id = _checked_entity_id(
                                candidate_row["source1_entity_id"],
                                source=f"candidate TSV line {line_number}",
                            )
                            candidate_list = candidate_row["candidate_entity_ids"] or ""
                            candidate_ids = list(
                                dict.fromkeys(
                                    candidate_id
                                    for candidate_id in candidate_list.split(",")
                                    if candidate_id
                                )
                            )
                            pair_batch.extend(
                                (source1_id, _checked_entity_id(
                                    candidate_id,
                                    source=f"candidate TSV line {line_number}",
                                ))
                                for candidate_id in candidate_ids
                            )
                            while len(pair_batch) >= pair_batch_size:
                                batch = pair_batch[:pair_batch_size]
                                del pair_batch[:pair_batch_size]
                                _write_pair_batch(
                                    connection,
                                    batch,
                                    writer=writer,
                                    vectorizers=vectorizers,
                                    idf=idf,
                                    normalize=normalize,
                                )
                        if pair_batch:
                            _write_pair_batch(
                                connection,
                                pair_batch,
                                writer=writer,
                                vectorizers=vectorizers,
                                idf=idf,
                                normalize=normalize,
                            )
                connection.close()
            except Exception:
                connection.close()
                raise
        os.replace(temporary_output, destination)
        temporary_output = None
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pairwise features for blocked business entity pairs."
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSV_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--idf-path", type=Path, default=None)
    parser.add_argument("--fit-idf-split", choices=("train", "test"), default="train")
    parser.add_argument("--working-directory", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--pair-batch-size", type=int, default=10_000)
    args = parser.parse_args()
    output_path = generate_pair_features(
        split=args.split,
        candidate_path=args.candidate_path,
        data_root=args.data_root,
        output_path=args.output,
        idf_path=args.idf_path,
        fit_idf_split=args.fit_idf_split,
        working_directory=args.working_directory,
        chunksize=args.chunk_size,
        pair_batch_size=args.pair_batch_size,
    )
    print(f"Wrote pair features to {output_path}")


if __name__ == "__main__":
    main()
