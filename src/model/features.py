from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
from dateutil import parser as date_parser


_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9 ]+")

ABBREV_MAP = {
    "W/O": "WITHOUT",
    "WO": "WITHOUT",
    "W": "WITH",
    "W/": "WITH",
    "W WO": "WITHOUT",
    "WOUT": "WITHOUT",
    "W CONTRAST": "WITH CONTRAST",
    "WO CONTRAST": "WITHOUT CONTRAST",
    "CNTRST": "CONTRAST",
    "CONTR": "CONTRAST",
    "C+": "CONTRAST",
    "CTA": "CT ANGIO",
    "MRA": "MR ANGIO",
    "C SPINE": "CERVICAL SPINE",
    "T SPINE": "THORACIC SPINE",
    "L SPINE": "LUMBAR SPINE",
    "CXR": "CHEST XRAY",
    "NONCON": "WITHOUT CONTRAST",
}

MODALITY_KEYWORDS = {
    "MRI": "MRI",
    "MR": "MRI",
    "CT": "CT",
    "XR": "XRAY",
    "X-RAY": "XRAY",
    "ULTRASOUND": "US",
    "US": "US",
    "PET": "PET",
    "NM": "NUCLEAR",
    "MAMMO": "MAMMO",
    "MAMMOGRAPHY": "MAMMO",
}

BODY_PARTS = {
    "BRAIN": "BRAIN",
    "HEAD": "BRAIN",
    "CHEST": "CHEST",
    "LUNG": "CHEST",
    "THORAX": "CHEST",
    "ABDOMEN": "ABDOMEN",
    "ABD": "ABDOMEN",
    "PELVIS": "PELVIS",
    "SPINE": "SPINE",
    "CERVICAL": "C-SPINE",
    "NECK": "C-SPINE",
    "THORACIC": "T-SPINE",
    "LUMBAR": "L-SPINE",
    "LOWBACK": "L-SPINE",
    "KNEE": "KNEE",
    "SHOULDER": "SHOULDER",
    "HIP": "HIP",
    "CARDIAC": "CARDIAC",
    "HEART": "CARDIAC",
    "CORONARY": "CORONARY",
    "CAROTID": "CAROTID",
}


@dataclass(frozen=True)
class PairRecord:
    case_id: str
    patient_id: str
    prior_study_id: str
    current_description: str
    current_date: str
    prior_description: str
    prior_date: str


def normalize_description(text: str) -> str:
    if not text:
        return ""

    cleaned = text.upper().strip()
    cleaned = cleaned.replace("WITHOUT CONTRAST", "WITHOUT CONTRAST")
    for src, dst in ABBREV_MAP.items():
        cleaned = cleaned.replace(src, dst)

    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def parse_date_safe(value: str) -> date | None:
    if not value:
        return None
    try:
        return date_parser.parse(value).date()
    except Exception:
        return None


def token_set(text: str) -> set[str]:
    if not text:
        return set()
    return set(normalize_description(text).split())


def extract_modality(text: str) -> str:
    tokens = token_set(text)
    for key, value in MODALITY_KEYWORDS.items():
        if key in tokens:
            return value
    return "UNKNOWN"


def extract_body_parts(text: str) -> set[str]:
    tokens = token_set(text)
    found: set[str] = set()
    for key, value in BODY_PARTS.items():
        if key in tokens:
            found.add(value)
    return found


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def overlap_coefficient(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(min(len(tokens_a), len(tokens_b)))


def build_pair_text(current_description: str, prior_description: str) -> str:
    curr = normalize_description(current_description)
    prior = normalize_description(prior_description)
    return f"CURRENT {curr} PRIOR {prior}"


def days_between(current_date: str, prior_date: str) -> int:
    current = parse_date_safe(current_date)
    prior = parse_date_safe(prior_date)
    if current is None or prior is None:
        return -1
    return (current - prior).days


def build_structured_features(records: Iterable[PairRecord]) -> np.ndarray:
    record_list = list(records)
    if not record_list:
        return np.zeros((0, 18), dtype=np.float32)

    total = len(record_list)
    curr_norm_list: list[str] = []
    prior_norm_list: list[str] = []
    curr_tokens_list: list[set[str]] = []
    prior_tokens_list: list[set[str]] = []
    curr_modality_list: list[str] = []
    prior_modality_list: list[str] = []
    curr_body_list: list[set[str]] = []
    prior_body_list: list[set[str]] = []
    delta_days_list: list[int] = []
    abs_gap_years_list: list[float] = []
    prior_is_before_list: list[float] = []

    case_to_indices: dict[str, list[int]] = {}
    recency_norm = np.zeros(total, dtype=np.float32)
    recency_is_most_recent = np.zeros(total, dtype=np.float32)
    case_size = np.ones(total, dtype=np.float32)
    same_modality_ratio = np.zeros(total, dtype=np.float32)
    same_body_ratio = np.zeros(total, dtype=np.float32)

    for idx, rec in enumerate(record_list):
        curr_norm = normalize_description(rec.current_description)
        prior_norm = normalize_description(rec.prior_description)

        curr_tokens = set(curr_norm.split())
        prior_tokens = set(prior_norm.split())

        curr_modality = extract_modality(curr_norm)
        prior_modality = extract_modality(prior_norm)

        curr_body = extract_body_parts(curr_norm)
        prior_body = extract_body_parts(prior_norm)

        delta_days = days_between(rec.current_date, rec.prior_date)
        abs_gap_years = 0.0 if delta_days == -1 else min(abs(delta_days) / 365.25, 30.0)
        prior_is_before = 0.0 if delta_days == -1 else (1.0 if delta_days >= 0 else 0.0)

        curr_norm_list.append(curr_norm)
        prior_norm_list.append(prior_norm)
        curr_tokens_list.append(curr_tokens)
        prior_tokens_list.append(prior_tokens)
        curr_modality_list.append(curr_modality)
        prior_modality_list.append(prior_modality)
        curr_body_list.append(curr_body)
        prior_body_list.append(prior_body)
        delta_days_list.append(delta_days)
        abs_gap_years_list.append(float(abs_gap_years))
        prior_is_before_list.append(float(prior_is_before))

        case_to_indices.setdefault(rec.case_id, []).append(idx)

    for indices in case_to_indices.values():
        size = float(len(indices))
        same_modality_count = 0.0
        same_body_count = 0.0
        dated_indices: list[tuple[int, int]] = []

        for idx in indices:
            if (
                curr_modality_list[idx] == prior_modality_list[idx]
                and curr_modality_list[idx] != "UNKNOWN"
            ):
                same_modality_count += 1.0

            if curr_body_list[idx] and prior_body_list[idx] and (curr_body_list[idx] & prior_body_list[idx]):
                same_body_count += 1.0

            if delta_days_list[idx] >= 0:
                dated_indices.append((idx, delta_days_list[idx]))

        for idx in indices:
            case_size[idx] = size
            same_modality_ratio[idx] = same_modality_count / size
            same_body_ratio[idx] = same_body_count / size

        if dated_indices:
            dated_indices.sort(key=lambda x: x[1])
            n = len(dated_indices)
            for rank, (idx, _) in enumerate(dated_indices):
                if n == 1:
                    recency_norm[idx] = 1.0
                else:
                    recency_norm[idx] = 1.0 - (rank / (n - 1))
                if rank == 0:
                    recency_is_most_recent[idx] = 1.0

    rows: list[list[float]] = []

    for idx in range(total):
        curr_norm = curr_norm_list[idx]
        prior_norm = prior_norm_list[idx]
        curr_tokens = curr_tokens_list[idx]
        prior_tokens = prior_tokens_list[idx]
        curr_modality = curr_modality_list[idx]
        prior_modality = prior_modality_list[idx]
        curr_body = curr_body_list[idx]
        prior_body = prior_body_list[idx]

        curr_has_contrast = 1.0 if "CONTRAST" in curr_tokens else 0.0
        prior_has_contrast = 1.0 if "CONTRAST" in prior_tokens else 0.0
        curr_has_without = 1.0 if "WITHOUT" in curr_tokens else 0.0
        prior_has_without = 1.0 if "WITHOUT" in prior_tokens else 0.0
        contrast_alignment = 1.0 if (curr_has_contrast == prior_has_contrast) else 0.0
        without_alignment = 1.0 if (curr_has_without == prior_has_without) else 0.0

        row = [
            1.0 if curr_norm == prior_norm and curr_norm else 0.0,
            1.0 if curr_modality == prior_modality and curr_modality != "UNKNOWN" else 0.0,
            1.0 if curr_modality != prior_modality else 0.0,
            1.0 if curr_body and prior_body and bool(curr_body & prior_body) else 0.0,
            float(jaccard_similarity(curr_tokens, prior_tokens)),
            float(overlap_coefficient(curr_tokens, prior_tokens)),
            float(abs_gap_years_list[idx]),
            float(prior_is_before_list[idx]),
            float(len(curr_tokens)),
            float(len(prior_tokens)),
            float(curr_has_contrast),
            float(prior_has_contrast),
            float(contrast_alignment),
            float(without_alignment),
            float(np.log1p(case_size[idx])),
            float(same_modality_ratio[idx]),
            float(same_body_ratio[idx]),
            float(recency_norm[idx]),
            float(recency_is_most_recent[idx]),
        ]
        rows.append(row)

    return np.asarray(rows, dtype=np.float32)
