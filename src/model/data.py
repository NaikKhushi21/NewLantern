from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.model.features import PairRecord


@dataclass(frozen=True)
class LabeledPair:
    record: PairRecord
    label: int


def load_dataset(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_truth_map(payload: dict) -> dict[tuple[str, str], int]:
    truth_map: dict[tuple[str, str], int] = {}
    for row in payload.get("truth", []):
        case_id = str(row["case_id"])
        study_id = str(row["study_id"])
        label = 1 if bool(row.get("is_relevant_to_current", False)) else 0
        truth_map[(case_id, study_id)] = label
    return truth_map


def flatten_labeled_pairs(payload: dict) -> list[LabeledPair]:
    truth_map = build_truth_map(payload)
    pairs: list[LabeledPair] = []

    for case in payload.get("cases", []):
        case_id = str(case["case_id"])
        current = case["current_study"]

        for prior in case.get("prior_studies", []):
            study_id = str(prior["study_id"])
            label = truth_map.get((case_id, study_id))
            if label is None:
                continue

            pairs.append(
                LabeledPair(
                    record=PairRecord(
                        case_id=case_id,
                        patient_id=str(case.get("patient_id", "")),
                        prior_study_id=study_id,
                        current_description=str(current.get("study_description", "")),
                        current_date=str(current.get("study_date", "")),
                        prior_description=str(prior.get("study_description", "")),
                        prior_date=str(prior.get("study_date", "")),
                    ),
                    label=label,
                )
            )

    return pairs


def flatten_unlabeled_pairs(payload: dict) -> list[PairRecord]:
    pairs: list[PairRecord] = []

    for case in payload.get("cases", []):
        case_id = str(case["case_id"])
        current = case["current_study"]
        for prior in case.get("prior_studies", []):
            pairs.append(
                PairRecord(
                    case_id=case_id,
                    patient_id=str(case.get("patient_id", "")),
                    prior_study_id=str(prior["study_id"]),
                    current_description=str(current.get("study_description", "")),
                    current_date=str(current.get("study_date", "")),
                    prior_description=str(prior.get("study_description", "")),
                    prior_date=str(prior.get("study_date", "")),
                )
            )

    return pairs
