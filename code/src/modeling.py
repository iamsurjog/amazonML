"""Precision-weighted entity matcher training and threshold tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.pair_features import PAIR_FEATURE_COLUMNS
from src.preprocessing import DEFAULT_TSV_ROOT


PROJECT_ROOT = DEFAULT_TSV_ROOT.parents[1]
DEFAULT_TRAIN_FEATURES_PATH = PROJECT_ROOT / "pair_features_train.tsv"
DEFAULT_GROUND_TRUTH_PATH = (
    DEFAULT_TSV_ROOT / "train" / "train_ground_truth.tsv"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "entity_match_model.joblib"
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")

MODEL_FEATURE_COLUMNS = tuple(PAIR_FEATURE_COLUMNS[3:]) + ("candidate_is_source3",)
NUMERIC_FEATURE_COLUMNS = tuple(PAIR_FEATURE_COLUMNS[3:])


@dataclass(frozen=True)
class TrainingSummary:
    model_path: Path
    threshold: float
    validation_macro_f0_5: float
    candidate_recall: float
    train_pair_count: int
    validation_pair_count: int
    train_positive_count: int
    validation_positive_count: int
    sampled_train_pair_count: int


class _StratifiedReservoir:
    """Keep a bounded uniform sample per label using random priorities."""

    def __init__(self, per_class_capacity: int, *, seed: int) -> None:
        self.capacity = per_class_capacity
        self.rng = np.random.default_rng(seed)
        self.priorities = [np.empty(0, dtype=np.float64) for _ in range(2)]
        self.features = [np.empty((0, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32) for _ in range(2)]

    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        for label in (0, 1):
            class_rows = features[labels == label]
            if len(class_rows) == 0:
                continue
            new_priorities = self.rng.random(len(class_rows))
            combined_priorities = np.concatenate((self.priorities[label], new_priorities))
            combined_features = np.concatenate((self.features[label], class_rows), axis=0)
            if len(combined_priorities) > self.capacity:
                selected = np.argpartition(combined_priorities, self.capacity - 1)[
                    : self.capacity
                ]
                combined_priorities = combined_priorities[selected]
                combined_features = combined_features[selected]
            self.priorities[label] = combined_priorities
            self.features[label] = combined_features

    def arrays(self, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self.features[0]) == 0 or len(self.features[1]) == 0:
            raise ValueError(
                "Training features must include at least one candidate positive and negative"
            )
        features = np.concatenate(self.features, axis=0)
        labels = np.concatenate(
            (
                np.zeros(len(self.features[0]), dtype=np.uint8),
                np.ones(len(self.features[1]), dtype=np.uint8),
            )
        )
        order = np.random.default_rng(seed).permutation(len(labels))
        return features[order], labels[order]


def _checked_id(value: object, *, description: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{description} is empty")
    entity_id = str(value)
    if not entity_id.strip() or any(c in entity_id for c in (",", "\t", "\r", "\n")):
        raise ValueError(f"{description} is empty or contains an output delimiter: {entity_id!r}")
    return entity_id


def _validation_group(source1_id: str, validation_fraction: float) -> bool:
    digest = hashlib.blake2b(
        source1_id.encode("utf-8"),
        digest_size=8,
        person=b"er-val-split",
    ).digest()
    percentile = int.from_bytes(digest, "little") % 10_000
    return percentile < int(validation_fraction * 10_000)


def _build_ground_truth_store(
    connection: sqlite3.Connection,
    ground_truth_path: Path,
    *,
    validation_fraction: float,
    chunksize: int,
) -> tuple[int, int]:
    connection.execute(
        "CREATE TABLE ground_truth ("
        "source1_entity_id TEXT PRIMARY KEY, matched_entity_ids TEXT NOT NULL, "
        "truth_count INTEGER NOT NULL, is_validation INTEGER NOT NULL) WITHOUT ROWID"
    )
    total_entities = 0
    total_truth_links = 0
    buffer: list[tuple[str, str, int, int]] = []

    with ground_truth_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Ground-truth TSV is empty or has no header")
        missing = [column for column in GROUND_TRUTH_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Ground-truth TSV is missing required columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            source1_id = _checked_id(
                row.get("source1_entity_id"),
                description=f"ground-truth line {line_number} source1_entity_id",
            )
            raw_matches = row.get("matched_entity_ids") or ""
            matched_ids = list(dict.fromkeys(match_id for match_id in raw_matches.split(",") if match_id))
            matched_ids = [
                _checked_id(match_id, description=f"ground-truth line {line_number} match ID")
                for match_id in matched_ids
            ]
            is_validation = int(_validation_group(source1_id, validation_fraction))
            buffer.append(
                (source1_id, ",".join(matched_ids), len(matched_ids), is_validation)
            )
            total_entities += 1
            total_truth_links += len(matched_ids)
            if len(buffer) >= chunksize:
                _flush_ground_truth(connection, buffer)
                buffer.clear()
    if buffer:
        _flush_ground_truth(connection, buffer)
    return total_entities, total_truth_links


def _flush_ground_truth(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, int, int]],
) -> None:
    try:
        with connection:
            connection.executemany(
                "INSERT INTO ground_truth "
                "(source1_entity_id, matched_entity_ids, truth_count, is_validation) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
    except sqlite3.IntegrityError as error:
        raise ValueError("Ground-truth source1_entity_id values must be unique") from error


def _make_batch_label_context(
    connection: sqlite3.Connection,
    source1_ids: Sequence[str],
) -> dict[str, tuple[set[str], bool]]:
    connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS batch_source1_ids ("
        "source1_entity_id TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    unique_ids = list(dict.fromkeys(source1_ids))
    with connection:
        connection.execute("DELETE FROM batch_source1_ids")
        connection.executemany(
            "INSERT INTO batch_source1_ids (source1_entity_id) VALUES (?)",
            ((source1_id,) for source1_id in unique_ids),
        )
    context: dict[str, tuple[set[str], bool]] = {}
    for source1_id, matched_entity_ids, is_validation in connection.execute(
        "SELECT g.source1_entity_id, g.matched_entity_ids, g.is_validation "
        "FROM batch_source1_ids AS b "
        "LEFT JOIN ground_truth AS g USING (source1_entity_id)"
    ):
        if matched_entity_ids is None:
            raise ValueError(f"No training ground truth for Source 1 ID {source1_id!r}")
        context[source1_id] = (
            set(matched_entity_ids.split(",")) if matched_entity_ids else set(),
            bool(is_validation),
        )
    if len(context) != len(unique_ids):
        raise RuntimeError("Ground-truth join did not return every Source 1 ID")
    return context


def prepare_model_features(frame: pd.DataFrame) -> np.ndarray:
    """Build the numeric feature matrix expected by the saved matcher bundle."""
    missing = sorted(set(PAIR_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Pair feature frame is missing required columns: {missing}")
    invalid_sources = ~frame["candidate_source"].isin(("source2", "source3"))
    if bool(invalid_sources.any()):
        bad_source = frame.loc[invalid_sources, "candidate_source"].iloc[0]
        raise ValueError(f"Unexpected candidate_source value: {bad_source!r}")
    numeric = frame.loc[:, NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    source3_flag = frame["candidate_source"].eq("source3").to_numpy(dtype=np.float32)
    return np.column_stack((numeric, source3_flag))


def _read_feature_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
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


def _labeled_feature_chunk(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_columns = set(PAIR_FEATURE_COLUMNS)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Pair feature TSV is missing required columns: {missing}")
    source1_ids = [
        _checked_id(value, description="pair-feature source1_entity_id")
        for value in frame["source1_entity_id"].tolist()
    ]
    candidate_ids = [
        _checked_id(value, description="pair-feature candidate_entity_id")
        for value in frame["candidate_entity_id"].tolist()
    ]
    context = _make_batch_label_context(connection, source1_ids)
    labels = np.fromiter(
        (int(candidate_id in context[source1_id][0]) for source1_id, candidate_id in zip(source1_ids, candidate_ids, strict=True)),
        dtype=np.uint8,
        count=len(frame),
    )
    validation_mask = np.fromiter(
        (context[source1_id][1] for source1_id in source1_ids),
        dtype=bool,
        count=len(frame),
    )
    return prepare_model_features(frame), labels, validation_mask, np.asarray(source1_ids, dtype=object)


def _entity_f0_5(true_positive: int, predicted: int, truth_count: int) -> float:
    if truth_count == 0:
        return 1.0 if predicted == 0 else 0.0
    if predicted == 0:
        return 0.0
    return 1.25 * true_positive / (predicted + 0.25 * truth_count)


def _macro_scores_for_thresholds(
    connection: sqlite3.Connection,
    truth_counts: dict[str, int],
    thresholds: Sequence[float],
) -> dict[float, float]:
    if not thresholds:
        return {}
    predicted_counts = dict.fromkeys(truth_counts, 0)
    true_positives = dict.fromkeys(truth_counts, 0)
    score_sum = float(sum(count == 0 for count in truth_counts.values()))
    thresholds_descending = sorted(set(float(value) for value in thresholds), reverse=True)
    scores: dict[float, float] = {}
    rows = iter(
        connection.execute(
            "SELECT source1_entity_id, probability, label "
            "FROM validation_scores ORDER BY probability DESC"
        )
    )
    pending = next(rows, None)

    for threshold in thresholds_descending:
        while pending is not None and float(pending[1]) >= threshold:
            source1_id, _, label = pending
            if source1_id not in truth_counts:
                raise ValueError(
                    f"Validation candidate has no held-out ground truth: {source1_id!r}"
                )
            truth_count = truth_counts[source1_id]
            old_f_score = _entity_f0_5(
                true_positives[source1_id],
                predicted_counts[source1_id],
                truth_count,
            )
            predicted_counts[source1_id] += 1
            if label:
                true_positives[source1_id] += 1
            new_f_score = _entity_f0_5(
                true_positives[source1_id],
                predicted_counts[source1_id],
                truth_count,
            )
            score_sum += new_f_score - old_f_score
            pending = next(rows, None)
        scores[threshold] = score_sum / len(truth_counts) if truth_counts else 0.0
    return scores


def _tune_threshold(connection: sqlite3.Connection, truth_counts: dict[str, int]) -> tuple[float, float]:
    coarse_thresholds = [index / 20.0 for index in range(21)]
    coarse_scores = _macro_scores_for_thresholds(
        connection,
        truth_counts,
        coarse_thresholds,
    )
    coarse_best = max(coarse_scores, key=lambda threshold: (coarse_scores[threshold], threshold))
    fine_thresholds = {
        round(min(1.0, max(0.0, coarse_best + offset / 200.0)), 4)
        for offset in range(-10, 11)
    }
    fine_scores = _macro_scores_for_thresholds(connection, truth_counts, sorted(fine_thresholds))
    all_scores = {**coarse_scores, **fine_scores}
    best_threshold = max(all_scores, key=lambda threshold: (all_scores[threshold], threshold))
    return best_threshold, all_scores[best_threshold]


def select_matches_above_threshold(
    all_source1_ids: Iterable[str],
    pair_source1_ids: Sequence[str],
    candidate_ids: Sequence[str],
    probabilities: Sequence[float],
    *,
    threshold: float,
    max_matches_per_source1: int | None = None,
) -> dict[str, list[str]]:
    """Select links at the tuned threshold, retaining empty lists for singletons."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not (len(pair_source1_ids) == len(candidate_ids) == len(probabilities)):
        raise ValueError("Pair ID and probability arrays must have equal lengths")
    if max_matches_per_source1 is not None and max_matches_per_source1 <= 0:
        raise ValueError("max_matches_per_source1 must be positive or None")

    selected: dict[str, list[tuple[float, str]]] = {
        str(source1_id): [] for source1_id in all_source1_ids
    }
    for source1_id, candidate_id, probability in zip(
        pair_source1_ids,
        candidate_ids,
        probabilities,
        strict=True,
    ):
        score = float(probability)
        if score >= threshold:
            selected.setdefault(str(source1_id), []).append((score, str(candidate_id)))

    matches: dict[str, list[str]] = {}
    for source1_id, scored_candidates in selected.items():
        unique_candidates: dict[str, float] = {}
        for score, candidate_id in scored_candidates:
            unique_candidates[candidate_id] = max(
                score,
                unique_candidates.get(candidate_id, float("-inf")),
            )
        ranked = sorted(unique_candidates.items(), key=lambda item: (-item[1], item[0]))
        if max_matches_per_source1 is not None:
            ranked = ranked[:max_matches_per_source1]
        matches[source1_id] = [candidate_id for candidate_id, _ in ranked]
    return matches


def train_precision_weighted_matcher(
    *,
    features_path: str | Path = DEFAULT_TRAIN_FEATURES_PATH,
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    working_directory: str | Path | None = None,
    chunksize: int = 50_000,
    max_training_samples_per_class: int = 250_000,
    validation_fraction: float = 0.2,
    max_iter: int = 160,
    seed: int = 42,
) -> TrainingSummary:
    """Train a histogram gradient booster and tune its threshold on held-out entities.

    Validation is split by Source 1 ID, and the tuned metric is the macro average
    of per-entity F_0.5 scores. Sampling is stratified and bounded so training
    does not require loading the full pair-feature table into RAM.
    """
    if chunksize <= 0 or max_training_samples_per_class <= 0 or max_iter <= 0:
        raise ValueError("chunksize, sample capacity, and max_iter must be positive")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as error:
        raise RuntimeError("Classifier training requires scikit-learn") from error

    feature_file = Path(features_path)
    truth_file = Path(ground_truth_path)
    destination = Path(model_path)
    for path, label in ((feature_file, "Pair-feature TSV"), (truth_file, "Ground-truth TSV")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() in (feature_file.resolve(), truth_file.resolve()):
        raise ValueError("model_path must not overwrite an input TSV")
    work_dir = Path(working_directory) if working_directory is not None else destination.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    validation_candidate_count = 0
    validation_positive_count = 0
    train_pair_count = 0
    train_positive_count = 0
    total_candidate_count = 0
    total_candidate_positive_count = 0
    temporary_model: Path | None = None

    with tempfile.TemporaryDirectory(prefix="entity-matcher-training-", dir=work_dir) as temp_dir:
        database_path = Path(temp_dir) / "training.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA cache_size = -65536")
            total_entities, total_truth_links = _build_ground_truth_store(
                connection,
                truth_file,
                validation_fraction=validation_fraction,
                chunksize=chunksize,
            )
            validation_truth_counts = {
                source1_id: int(truth_count)
                for source1_id, truth_count in connection.execute(
                    "SELECT source1_entity_id, truth_count FROM ground_truth "
                    "WHERE is_validation = 1"
                )
            }
            if not validation_truth_counts:
                raise ValueError("The selected validation split contains no Source 1 entities")

            reservoir = _StratifiedReservoir(
                max_training_samples_per_class,
                seed=seed,
            )
            for frame in _read_feature_chunks(feature_file, chunksize):
                features, labels, validation_mask, _ = _labeled_feature_chunk(
                    connection,
                    frame,
                )
                total_candidate_count += len(frame)
                total_candidate_positive_count += int(labels.sum())
                validation_candidate_count += int(validation_mask.sum())
                validation_positive_count += int(labels[validation_mask].sum())
                train_mask = ~validation_mask
                train_pair_count += int(train_mask.sum())
                train_positive_count += int(labels[train_mask].sum())
                reservoir.update(features[train_mask], labels[train_mask])

            train_features, train_labels = reservoir.arrays(seed=seed)
            sampled_train_pair_count = len(train_labels)
            classifier = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.08,
                max_iter=max_iter,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=False,
                random_state=seed,
            )
            classifier.fit(train_features, train_labels)
            del train_features, train_labels, reservoir

            connection.execute(
                "CREATE TABLE validation_scores ("
                "source1_entity_id TEXT NOT NULL, probability REAL NOT NULL, label INTEGER NOT NULL)"
            )
            for frame in _read_feature_chunks(feature_file, chunksize):
                features, labels, validation_mask, source1_ids = _labeled_feature_chunk(
                    connection,
                    frame,
                )
                if not bool(validation_mask.any()):
                    continue
                validation_features = features[validation_mask]
                probabilities = classifier.predict_proba(validation_features)[:, 1]
                validation_ids = source1_ids[validation_mask]
                validation_labels = labels[validation_mask]
                score_rows = [
                    (str(source1_id), float(probability), int(label))
                    for source1_id, probability, label in zip(
                        validation_ids,
                        probabilities,
                        validation_labels,
                        strict=True,
                    )
                ]
                with connection:
                    connection.executemany(
                        "INSERT INTO validation_scores "
                        "(source1_entity_id, probability, label) VALUES (?, ?, ?)",
                        score_rows,
                    )
            connection.execute(
                "CREATE INDEX validation_probability_idx "
                "ON validation_scores (probability DESC)"
            )
            threshold, validation_macro_f0_5 = _tune_threshold(
                connection,
                validation_truth_counts,
            )

            candidate_recall = (
                total_candidate_positive_count / total_truth_links
                if total_truth_links
                else 0.0
            )
            bundle = {
                "model": classifier,
                "threshold": threshold,
                "feature_columns": MODEL_FEATURE_COLUMNS,
                "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
                "validation_macro_f0_5": validation_macro_f0_5,
                "validation_fraction": validation_fraction,
                "validation_entity_count": len(validation_truth_counts),
                "validation_pair_count": validation_candidate_count,
                "validation_positive_count": validation_positive_count,
                "train_pair_count": train_pair_count,
                "train_positive_count": train_positive_count,
                "sampled_train_pair_count": sampled_train_pair_count,
                "candidate_recall": candidate_recall,
                "total_ground_truth_links": total_truth_links,
                "total_ground_truth_entities": total_entities,
                "training_features_path": str(feature_file),
            }
        finally:
            connection.close()

    try:
        import joblib

        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".joblib",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_model = Path(temporary_file.name)
        joblib.dump(bundle, temporary_model, compress=3)
        os.replace(temporary_model, destination)
        temporary_model = None
    finally:
        if temporary_model is not None:
            temporary_model.unlink(missing_ok=True)

    return TrainingSummary(
        model_path=destination,
        threshold=threshold,
        validation_macro_f0_5=validation_macro_f0_5,
        candidate_recall=candidate_recall,
        train_pair_count=train_pair_count,
        validation_pair_count=validation_candidate_count,
        train_positive_count=train_positive_count,
        validation_positive_count=validation_positive_count,
        sampled_train_pair_count=int(bundle["sampled_train_pair_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the precision-weighted business entity matcher."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_TRAIN_FEATURES_PATH)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--working-directory", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-samples-per-class", type=int, default=250_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = train_precision_weighted_matcher(
        features_path=args.features,
        ground_truth_path=args.ground_truth,
        model_path=args.model,
        working_directory=args.working_directory,
        chunksize=args.chunk_size,
        max_training_samples_per_class=args.max_samples_per_class,
        validation_fraction=args.validation_fraction,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    print(f"Saved model: {summary.model_path}")
    print(f"Tuned probability threshold: {summary.threshold:.4f}")
    print(f"Held-out macro F0.5: {summary.validation_macro_f0_5:.6f}")
    print(f"Candidate-pair recall: {summary.candidate_recall:.6f}")
    print(
        "Training rows: "
        f"{summary.train_pair_count:,} (positive {summary.train_positive_count:,}); "
        f"validation rows: {summary.validation_pair_count:,} "
        f"(positive {summary.validation_positive_count:,})"
    )


if __name__ == "__main__":
    main()
