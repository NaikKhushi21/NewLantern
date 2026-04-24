from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.model.data import build_truth_map, flatten_unlabeled_pairs, load_dataset
from src.model.infer import load_bundle, predict_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local model artifacts against labeled JSON")
    parser.add_argument("--data", default="relevant_priors_public.json")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--chunk-size", type=int, default=0, help="Optional chunking to simulate batched requests")
    return parser.parse_args()


def accuracy_from_pairs(
    predictions: list[tuple[str, str, bool]],
    truth_map: dict[tuple[str, str], int],
) -> tuple[float, int, int, int]:
    correct = 0
    incorrect = 0
    missing = 0

    pred_map: dict[tuple[str, str], bool] = {(c, s): p for c, s, p in predictions}

    for key, label in truth_map.items():
        pred = pred_map.get(key)
        if pred is None:
            missing += 1
            incorrect += 1
            continue
        if bool(label) == bool(pred):
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect
    acc = 0.0 if total == 0 else correct / total
    return acc, correct, incorrect, missing


def main() -> None:
    args = parse_args()

    payload = load_dataset(args.data)
    truth_map = build_truth_map(payload)
    pairs = flatten_unlabeled_pairs(payload)

    bundle = load_bundle(args.model)

    started = time.perf_counter()
    predictions: list[tuple[str, str, bool]] = []

    if args.chunk_size and args.chunk_size > 0:
        for i in range(0, len(pairs), args.chunk_size):
            chunk = pairs[i : i + args.chunk_size]
            chunk_preds = predict_pairs(chunk, bundle)
            predictions.extend((row.case_id, row.prior_study_id, pred) for row, pred in zip(chunk, chunk_preds))
    else:
        preds = predict_pairs(pairs, bundle)
        predictions = [(row.case_id, row.prior_study_id, pred) for row, pred in zip(pairs, preds)]

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    acc, correct, incorrect, missing = accuracy_from_pairs(predictions, truth_map)

    result = {
        "pairs_in_data": len(truth_map),
        "predictions_returned": len(predictions),
        "accuracy": acc,
        "correct": correct,
        "incorrect": incorrect,
        "missing_predictions": missing,
        "elapsed_ms": elapsed_ms,
    }

    print(json.dumps(result, indent=2))

    if len(predictions) != len(truth_map):
        raise SystemExit("Prediction count mismatch with truth labels")


if __name__ == "__main__":
    main()
