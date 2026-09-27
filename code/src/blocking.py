"""Streaming TF-IDF + MinHash-LSH candidate generation."""

from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import (
    DEFAULT_TSV_ROOT,
    SourceSplit,
    iter_source_tsv,
)


DEFAULT_CANDIDATE_PATH = DEFAULT_TSV_ROOT.parents[1] / "candidate_pairs.tsv"
OUTPUT_COLUMNS = ("source1_entity_id", "candidate_entity_ids")


def _identity_feature_hash(value: int) -> int:
    """Use already-hashed vectorizer feature IDs as MinHash input values."""
    return int(value)


def _country_key(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _entity_id(value: object, *, source: str) -> str:
    if value is None or bool(pd.isna(value)):
        raise ValueError(f"{source} contains an empty entity_id")
    entity_id = str(value)
    if not entity_id.strip():
        raise ValueError(f"{source} contains an empty entity_id")
    if any(character in entity_id for character in (",", "\t", "\r", "\n")):
        raise ValueError(
            f"{source} entity_id {entity_id!r} contains a delimiter used by the output format"
        )
    return entity_id


def _new_vectorizer(vectorizer_class: Any, *, n_features: int, ngram_range: tuple[int, int]) -> Any:
    return vectorizer_class(
        analyzer="char_wb",
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        binary=True,
        lowercase=False,
        norm=None,
        dtype=np.float32,
    )


def _fit_streaming_idf(
    source_paths: dict[str, Path],
    *,
    chunksize: int,
    n_features: int,
    ngram_range: tuple[int, int],
    vectorizer_class: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, int]]:
    """Count document frequencies in chunks; derive smoothed IDF without a corpus matrix."""
    field_columns = {
        "name": "business_name_normalized",
        "address": "business_address_normalized",
    }
    vectorizers = {
        field: _new_vectorizer(
            vectorizer_class,
            n_features=n_features,
            ngram_range=ngram_range,
        )
        for field in field_columns
    }
    document_frequencies = {
        field: np.zeros(n_features, dtype=np.int64) for field in field_columns
    }
    document_count = 0
    row_counts: dict[str, int] = {}

    for source_name, path in source_paths.items():
        rows_for_source = 0
        for frame in iter_source_tsv(path, chunksize=chunksize, normalize=True):
            rows_for_source += len(frame)
            document_count += len(frame)
            for field, column in field_columns.items():
                texts = frame[column].fillna("").astype("string").tolist()
                matrix = vectorizers[field].transform(texts)
                document_frequencies[field] += np.asarray(
                    matrix.getnnz(axis=0), dtype=np.int64
                ).reshape(-1)
        row_counts[source_name] = rows_for_source

    idf = {
        field: np.log((1.0 + document_count) / (1.0 + frequencies)) + 1.0
        for field, frequencies in document_frequencies.items()
    }
    return vectorizers, idf, row_counts


def _top_tfidf_feature_ids(
    matrix: Any,
    row_index: int,
    idf: np.ndarray,
    *,
    limit: int,
    feature_offset: int,
) -> list[int]:
    """Select stable, high-IDF features from one binary TF-IDF document row."""
    start, end = matrix.indptr[row_index : row_index + 2]
    indices = matrix.indices[start:end]
    if len(indices) == 0:
        return []

    # HashingVectorizer uses binary term frequency here, so TF-IDF is TF * IDF.
    weights = matrix.data[start:end] * idf[indices]
    if len(indices) > limit:
        positions = np.argpartition(weights, -limit)[-limit:]
    else:
        positions = np.arange(len(indices))
    positions = sorted(
        (int(position) for position in positions),
        key=lambda position: (-float(weights[position]), int(indices[position])),
    )
    return [int(indices[position]) + feature_offset for position in positions]


def _packed_parts(*parts: bytes) -> bytes:
    return b"".join(len(part).to_bytes(4, "little") + part for part in parts)


def _exact_bucket_hash(country: str, field: bytes, value: str) -> bytes | None:
    if not value:
        return None
    payload = _packed_parts(b"exact", country.encode("utf-8"), field, value.encode("utf-8"))
    return hashlib.blake2b(payload, digest_size=16).digest()


def _record_buckets(
    *,
    country: str,
    name: str,
    address: str,
    name_feature_ids: list[int],
    address_feature_ids: list[int],
    minhash: Any,
    bands: int,
    rows_per_band: int,
) -> list[tuple[bytes, int]]:
    """Return (bucket hash, exact-match flag) pairs for one entity."""
    buckets: dict[bytes, int] = {}
    feature_ids = name_feature_ids + address_feature_ids
    if feature_ids:
        minhash.hashvalues.fill(np.iinfo(minhash.hashvalues.dtype).max)
        minhash.update_batch(feature_ids)
        signature = minhash.hashvalues
        country_bytes = country.encode("utf-8")
        for band in range(bands):
            start = band * rows_per_band
            band_values = np.asarray(
                signature[start : start + rows_per_band], dtype="<u8"
            ).tobytes()
            payload = _packed_parts(
                b"minhash",
                country_bytes,
                band.to_bytes(2, "little"),
                band_values,
            )
            key = hashlib.blake2b(payload, digest_size=16).digest()
            buckets[key] = 0

    for field, value in ((b"name", name), (b"address", address)):
        key = _exact_bucket_hash(country, field, value)
        if key is not None:
            buckets[key] = 1
    return list(buckets.items())


def _source_paths(split: SourceSplit, data_root: str | Path | None) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else DEFAULT_TSV_ROOT
    split_directory = root / split
    return {
        source_name: split_directory / f"{split}_{source_name}.tsv"
        for source_name in ("source1", "source2", "source3")
    }


def _insert_target_index(
    connection: sqlite3.Connection,
    *,
    source_paths: dict[str, Path],
    vectorizers: dict[str, Any],
    idf: dict[str, np.ndarray],
    chunksize: int,
    max_name_features: int,
    max_address_features: int,
    minhash: Any,
    bands: int,
    rows_per_band: int,
    feature_offset: int,
) -> int:
    target_row = 0
    record_buffer: list[tuple[int, str]] = []
    bucket_buffer: list[tuple[bytes, int, int]] = []
    flush_after_records = 2_000

    def flush() -> None:
        if not record_buffer:
            return
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO target_records (target_row, entity_id) VALUES (?, ?)",
                    record_buffer,
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO lsh_buckets "
                    "(bucket_hash, target_row, exact_match) VALUES (?, ?, ?)",
                    bucket_buffer,
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                "Source 2 and Source 3 entity_id values must be unique"
            ) from error
        record_buffer.clear()
        bucket_buffer.clear()

    for source_name in ("source2", "source3"):
        path = source_paths[source_name]
        for frame in iter_source_tsv(path, chunksize=chunksize, normalize=True):
            name_matrix = vectorizers["name"].transform(
                frame["business_name_normalized"].fillna("").astype("string").tolist()
            )
            address_matrix = vectorizers["address"].transform(
                frame["business_address_normalized"].fillna("").astype("string").tolist()
            )
            ids = frame["entity_id"].tolist()
            countries = frame["country"].tolist()
            names = frame["business_name_normalized"].tolist()
            addresses = frame["business_address_normalized"].tolist()

            for local_row in range(len(frame)):
                entity_id = _entity_id(ids[local_row], source=source_name)
                name = str(names[local_row])
                address = str(addresses[local_row])
                name_features = _top_tfidf_feature_ids(
                    name_matrix,
                    local_row,
                    idf["name"],
                    limit=max_name_features,
                    feature_offset=0,
                )
                address_features = _top_tfidf_feature_ids(
                    address_matrix,
                    local_row,
                    idf["address"],
                    limit=max_address_features,
                    feature_offset=feature_offset,
                )
                buckets = _record_buckets(
                    country=_country_key(countries[local_row]),
                    name=name,
                    address=address,
                    name_feature_ids=name_features,
                    address_feature_ids=address_features,
                    minhash=minhash,
                    bands=bands,
                    rows_per_band=rows_per_band,
                )
                record_buffer.append((target_row, entity_id))
                bucket_buffer.extend(
                    (bucket_hash, target_row, exact_match)
                    for bucket_hash, exact_match in buckets
                )
                target_row += 1
                if len(record_buffer) >= flush_after_records:
                    flush()
    flush()
    return target_row


def _write_source1_candidates(
    connection: sqlite3.Connection,
    *,
    source1_path: Path,
    output_stream: Any,
    vectorizers: dict[str, Any],
    idf: dict[str, np.ndarray],
    chunksize: int,
    max_name_features: int,
    max_address_features: int,
    minhash: Any,
    bands: int,
    rows_per_band: int,
    feature_offset: int,
    max_candidates_per_source1: int | None,
) -> int:
    writer = csv.writer(output_stream, delimiter="\t", lineterminator="\n")
    writer.writerow(OUTPUT_COLUMNS)
    connection.execute(
        "CREATE TEMP TABLE query_buckets ("
        "query_row INTEGER NOT NULL, bucket_hash BLOB NOT NULL, exact_match INTEGER NOT NULL, "
        "PRIMARY KEY (query_row, bucket_hash)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TEMP TABLE source1_seen (entity_id TEXT PRIMARY KEY) WITHOUT ROWID"
    )

    limit = max_candidates_per_source1 or 9_223_372_036_854_775_807
    candidate_sql = """
        WITH collisions AS (
            SELECT q.query_row,
                   b.target_row,
                   SUM(b.exact_match) AS exact_hits,
                   COUNT(*) - SUM(b.exact_match) AS lsh_votes
            FROM query_buckets AS q
            JOIN lsh_buckets AS b ON b.bucket_hash = q.bucket_hash
            GROUP BY q.query_row, b.target_row
        ), ranked AS (
            SELECT c.query_row,
                   t.entity_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.query_row
                       ORDER BY c.exact_hits DESC, c.lsh_votes DESC, t.entity_id ASC
                   ) AS candidate_rank
            FROM collisions AS c
            JOIN target_records AS t ON t.target_row = c.target_row
        )
        SELECT query_row, entity_id
        FROM ranked
        WHERE candidate_rank <= ?
        ORDER BY query_row, candidate_rank
    """

    written = 0
    for frame in iter_source_tsv(source1_path, chunksize=chunksize, normalize=True):
        ids = [
            _entity_id(value, source="source1")
            for value in frame["entity_id"].tolist()
        ]
        before_seen_insert = connection.total_changes
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO source1_seen (entity_id) VALUES (?)",
                ((entity_id,) for entity_id in ids),
            )
        if connection.total_changes - before_seen_insert != len(ids):
            raise ValueError("Source 1 contains duplicate entity_id values")

        name_matrix = vectorizers["name"].transform(
            frame["business_name_normalized"].fillna("").astype("string").tolist()
        )
        address_matrix = vectorizers["address"].transform(
            frame["business_address_normalized"].fillna("").astype("string").tolist()
        )
        countries = frame["country"].tolist()
        names = frame["business_name_normalized"].tolist()
        addresses = frame["business_address_normalized"].tolist()

        query_bucket_rows: list[tuple[int, bytes, int]] = []
        for local_row in range(len(frame)):
            name = str(names[local_row])
            address = str(addresses[local_row])
            name_features = _top_tfidf_feature_ids(
                name_matrix,
                local_row,
                idf["name"],
                limit=max_name_features,
                feature_offset=0,
            )
            address_features = _top_tfidf_feature_ids(
                address_matrix,
                local_row,
                idf["address"],
                limit=max_address_features,
                feature_offset=feature_offset,
            )
            buckets = _record_buckets(
                country=_country_key(countries[local_row]),
                name=name,
                address=address,
                name_feature_ids=name_features,
                address_feature_ids=address_features,
                minhash=minhash,
                bands=bands,
                rows_per_band=rows_per_band,
            )
            query_bucket_rows.extend(
                (local_row, bucket_hash, exact_match)
                for bucket_hash, exact_match in buckets
            )

        with connection:
            connection.execute("DELETE FROM query_buckets")
            connection.executemany(
                "INSERT OR IGNORE INTO query_buckets "
                "(query_row, bucket_hash, exact_match) VALUES (?, ?, ?)",
                query_bucket_rows,
            )

        candidates_by_row: list[list[str]] = [[] for _ in ids]
        for query_row, candidate_id in connection.execute(candidate_sql, (limit,)):
            candidates_by_row[query_row].append(candidate_id)
        for source1_id, candidate_ids in zip(ids, candidates_by_row, strict=True):
            # Defensive stable deduplication keeps the serialized candidate list clean.
            unique_ids = list(dict.fromkeys(candidate_ids))
            writer.writerow((source1_id, ",".join(unique_ids)))
            written += 1

    return written


def generate_candidate_pairs(
    split: SourceSplit = "test",
    *,
    data_root: str | Path | None = None,
    output_path: str | Path | None = None,
    working_directory: str | Path | None = None,
    chunksize: int = 10_000,
    n_features: int = 1 << 20,
    ngram_range: tuple[int, int] = (2, 5),
    max_name_features: int = 48,
    max_address_features: int = 48,
    num_perm: int = 32,
    bands: int = 16,
    max_candidates_per_source1: int | None = 500,
    seed: int = 1,
) -> Path:
    """Generate ``candidate_pairs.tsv`` using streamed TF-IDF and MinHash LSH.

    Character n-grams are hashed into fixed-size vectors. A first streaming pass
    computes smoothed IDF statistics over all three sources; the second pass uses
    the highest-weight name/address features to create MinHash signatures. The
    LSH bands are stored in a temporary SQLite index, so target records and the
    pairwise candidate space are never loaded as a full cross-join in memory.

    Country values are used as dynamic block keys (including unseen values such
    as France); the implementation contains no country allowlist. Every Source 1
    ID is written, with an empty candidate field when no block is hit.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    if n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    if min(ngram_range) <= 0 or ngram_range[0] > ngram_range[1]:
        raise ValueError(f"Invalid ngram_range: {ngram_range}")
    if max_name_features <= 0 or max_address_features <= 0:
        raise ValueError("Feature limits must be positive integers")
    if bands <= 0 or num_perm <= 0 or num_perm % bands:
        raise ValueError("num_perm must be a positive multiple of bands")
    if max_candidates_per_source1 is not None and max_candidates_per_source1 <= 0:
        raise ValueError("max_candidates_per_source1 must be positive or None")

    try:
        from datasketch import MinHash
        from sklearn.feature_extraction.text import HashingVectorizer
    except ImportError as error:
        raise RuntimeError(
            "Phase 2 requires scikit-learn and datasketch; install requirements.txt"
        ) from error

    source_paths = _source_paths(split, data_root)
    missing_files = [path for path in source_paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing source TSV file(s): " + ", ".join(str(path) for path in missing_files)
        )

    default_candidate_path = (
        DEFAULT_CANDIDATE_PATH
        if split == "test"
        else DEFAULT_CANDIDATE_PATH.with_name("candidate_pairs_train.tsv")
    )
    destination = Path(output_path) if output_path is not None else default_candidate_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() in {path.resolve() for path in source_paths.values()}:
        raise ValueError("output_path must not overwrite an input source TSV")

    work_dir = Path(working_directory) if working_directory is not None else destination.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    rows_per_band = num_perm // bands
    feature_offset = n_features
    output_tmp_path: Path | None = None

    vectorizers, idf, source_row_counts = _fit_streaming_idf(
        source_paths,
        chunksize=chunksize,
        n_features=n_features,
        ngram_range=ngram_range,
        vectorizer_class=HashingVectorizer,
    )
    minhash = MinHash(
        num_perm=num_perm,
        seed=seed,
        hashfunc=_identity_feature_hash,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="entity-resolution-lsh-", dir=work_dir) as temp_dir:
            index_path = Path(temp_dir) / "lsh_index.sqlite"
            connection = sqlite3.connect(index_path)
            try:
                connection.execute("PRAGMA journal_mode = OFF")
                connection.execute("PRAGMA synchronous = OFF")
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute("PRAGMA cache_size = -65536")
                connection.executescript(
                    "CREATE TABLE target_records ("
                    "target_row INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE);"
                    "CREATE TABLE lsh_buckets ("
                    "bucket_hash BLOB NOT NULL, target_row INTEGER NOT NULL, "
                    "exact_match INTEGER NOT NULL, "
                    "PRIMARY KEY (bucket_hash, target_row)) WITHOUT ROWID;"
                )

                _insert_target_index(
                    connection,
                    source_paths=source_paths,
                    vectorizers=vectorizers,
                    idf=idf,
                    chunksize=chunksize,
                    max_name_features=max_name_features,
                    max_address_features=max_address_features,
                    minhash=minhash,
                    bands=bands,
                    rows_per_band=rows_per_band,
                    feature_offset=feature_offset,
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
                    output_tmp_path = Path(output_stream.name)
                    rows_written = _write_source1_candidates(
                        connection,
                        source1_path=source_paths["source1"],
                        output_stream=output_stream,
                        vectorizers=vectorizers,
                        idf=idf,
                        chunksize=chunksize,
                        max_name_features=max_name_features,
                        max_address_features=max_address_features,
                        minhash=minhash,
                        bands=bands,
                        rows_per_band=rows_per_band,
                        feature_offset=feature_offset,
                        max_candidates_per_source1=max_candidates_per_source1,
                    )
                connection.close()
            except Exception:
                connection.close()
                raise

        if rows_written != source_row_counts["source1"]:
            raise RuntimeError(
                "Candidate output row count mismatch: "
                f"wrote {rows_written}, expected {source_row_counts['source1']}"
            )
        os.replace(output_tmp_path, destination)
        output_tmp_path = None
    finally:
        if output_tmp_path is not None:
            output_tmp_path.unlink(missing_ok=True)

    return destination
