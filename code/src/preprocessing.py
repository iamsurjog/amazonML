"""TSV ingestion and text normalization for business entity resolution."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pandas as pd

try:
    from anyascii import anyascii as _anyascii
except ImportError:  # Keep the preprocessing module usable before dependencies are installed.
    _anyascii = None


SourceSplit = Literal["train", "test"]
SOURCE_NAMES = ("source1", "source2", "source3")
REQUIRED_SOURCE_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)

DEFAULT_TSV_ROOT = Path(__file__).resolve().parents[1] / "data" / "tsv"

_NAME_TOKEN_ALIASES = {
    "co": "company",
    "company": "company",
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "st": "saint",
}

_ADDRESS_TOKEN_ALIASES = {
    **_NAME_TOKEN_ALIASES,
    "apt": "apartment",
    "ave": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "bldg": "building",
    "building": "building",
    "ct": "court",
    "court": "court",
    "dr": "drive",
    "drive": "drive",
    "hwy": "highway",
    "highway": "highway",
    "ln": "lane",
    "lane": "lane",
    "pkwy": "parkway",
    "parkway": "parkway",
    "rd": "road",
    "road": "road",
    "st": "street",
    "street": "street",
    "ste": "suite",
    "suite": "suite",
}


def _coerce_text(value: object) -> str:
    """Return a string for a scalar input, treating null values as empty text."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        # Inputs are normally scalars; retain unusual values rather than failing here.
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_text(value: object, aliases: dict[str, str]) -> str:
    text = unicodedata.normalize("NFKC", _coerce_text(value)).casefold()
    if _anyascii is not None:
        # anyascii provides deterministic offline transliteration for non-Latin scripts.
        text = _anyascii(text)
    else:
        # Built-in fallback handles compatibility forms and Latin diacritics.
        text = "".join(
            character
            for character in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(character)
        )

    text = text.replace("&", " and ")
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    tokens = (aliases.get(token, token) for token in text.split())
    return " ".join(tokens)


def normalize_business_name(value: object) -> str:
    """Normalize a business name, including common legal-form abbreviations."""
    return _normalize_text(value, _NAME_TOKEN_ALIASES)


def normalize_business_address(value: object) -> str:
    """Normalize an address, including common street and unit abbreviations."""
    return _normalize_text(value, _ADDRESS_TOKEN_ALIASES)


def _validate_source_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Source TSV is missing required columns "
            f"{missing}; found columns {list(frame.columns)}"
        )


def normalize_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized name and address columns appended.

    Original columns are preserved for downstream feature engineering. Country is
    intentionally left untouched; no country-specific filters or assumptions are used.
    """
    _validate_source_frame(frame)
    normalized = frame.copy()
    normalized["business_name_normalized"] = (
        frame["business_name"]
        .astype("string")
        .fillna("")
        .map(normalize_business_name)
        .astype("string")
    )
    normalized["business_address_normalized"] = (
        frame["business_address"]
        .astype("string")
        .fillna("")
        .map(normalize_business_address)
        .astype("string")
    )
    return normalized


def read_source_tsv(
    path: str | Path,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Read one source TSV while preserving IDs and empty fields as strings."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source TSV does not exist: {source_path}")

    frame = pd.read_csv(
        source_path,
        sep="\t",
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="error",
    )
    _validate_source_frame(frame)
    return normalize_source_frame(frame) if normalize else frame


def iter_source_tsv(
    path: str | Path,
    *,
    chunksize: int = 100_000,
    normalize: bool = True,
) -> Iterator[pd.DataFrame]:
    """Read and optionally normalize a source TSV in bounded-memory chunks."""
    if chunksize <= 0:
        raise ValueError(f"chunksize must be a positive integer; got {chunksize}")

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source TSV does not exist: {source_path}")

    reader = pd.read_csv(
        source_path,
        sep="\t",
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="error",
        chunksize=chunksize,
    )
    for frame in reader:
        _validate_source_frame(frame)
        yield normalize_source_frame(frame) if normalize else frame


def load_source_split(
    split: SourceSplit = "train",
    *,
    data_root: str | Path | None = None,
    normalize: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load Source 1, Source 2, and Source 3 for a train or test split.

    ``data_root`` should point to the directory containing the ``train`` and
    ``test`` subdirectories (by default, ``data/tsv`` in the project root).
    For datasets too large to keep in memory, use :func:`iter_source_tsv` per file.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")

    tsv_root = Path(data_root) if data_root is not None else DEFAULT_TSV_ROOT
    split_dir = tsv_root / split
    return {
        source_name: read_source_tsv(
            split_dir / f"{split}_{source_name}.tsv",
            normalize=normalize,
        )
        for source_name in SOURCE_NAMES
    }


def load_training_sources(
    *,
    data_root: str | Path | None = None,
    normalize: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load and preprocess the three training entity sources."""
    return load_source_split("train", data_root=data_root, normalize=normalize)
