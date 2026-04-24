from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.model.features import extract_modality, normalize_description
from src.model.features import PairRecord
from src.model.pipeline import FeatureArtifacts, build_feature_matrix_transform


@dataclass
class InferenceBundle:
    model: Any
    threshold: float
    threshold_config: dict[str, Any] | None
    feature_artifacts: FeatureArtifacts


def load_bundle(path: str | Path) -> InferenceBundle:
    payload = joblib.load(path)
    feature_artifacts = FeatureArtifacts(
        word_vectorizer=payload["word_vectorizer"],
        char_vectorizer=payload["char_vectorizer"],
    )
    return InferenceBundle(
        model=payload["model"],
        threshold=float(payload.get("threshold", 0.5)),
        threshold_config=payload.get("threshold_config"),
        feature_artifacts=feature_artifacts,
    )


def predict_pairs(records: list[PairRecord], bundle: InferenceBundle) -> list[bool]:
    if not records:
        return []

    x = build_feature_matrix_transform(records, bundle.feature_artifacts)

    if hasattr(bundle.model, "predict_proba"):
        scores = bundle.model.predict_proba(x)[:, 1]
        threshold_config = bundle.threshold_config or {}
        global_threshold = float(threshold_config.get("global", bundle.threshold))
        by_modality = threshold_config.get("by_modality") or {}

        if by_modality:
            labels: list[bool] = []
            for record, score in zip(records, scores):
                modality = extract_modality(normalize_description(record.current_description))
                threshold = float(by_modality.get(modality, global_threshold))
                labels.append(bool(score >= threshold))
            return labels

        return [bool(v) for v in (scores >= global_threshold)]

    raw = bundle.model.predict(x)
    return [bool(v) for v in np.asarray(raw).astype(int)]
