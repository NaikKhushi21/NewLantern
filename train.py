from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from src.model.data import flatten_labeled_pairs, load_dataset
from src.model.features import extract_modality, normalize_description
from src.model.pipeline import build_feature_matrix_fit, build_feature_matrix_transform


THRESH_GRID = np.linspace(0.15, 0.85, 71)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train relevant priors model")
    parser.add_argument("--data", default="relevant_priors_public.json", help="Path to labeled public JSON")
    parser.add_argument("--out", default="artifacts/model.joblib", help="Output artifact path")
    parser.add_argument("--metrics-out", default="artifacts/train_metrics.json", help="Metrics JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--mod-threshold-min-samples", type=int, default=300)
    parser.add_argument("--mod-threshold-shrink-k", type=float, default=1000.0)
    return parser.parse_args()


def tune_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_acc = -1.0

    for threshold in THRESH_GRID:
        pred = (probs >= threshold).astype(int)
        acc = accuracy_score(labels, pred)
        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)

    return best_threshold, best_acc


def config_key(cfg: dict) -> str:
    return f"C={cfg['c']},class_weight={cfg['class_weight']}"


def pick_groups(records: list) -> np.ndarray:
    groups = np.asarray([r.patient_id for r in records], dtype=object)
    if np.any(groups == ""):
        fallback = np.asarray([r.case_id for r in records], dtype=object)
        groups = np.where(groups == "", fallback, groups)
    return groups


def tune_modality_thresholds(
    modalities: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    global_threshold: float,
    min_samples: int,
    shrink_k: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for modality in sorted(set(modalities.tolist())):
        idx = np.where(modalities == modality)[0]
        if len(idx) < min_samples:
            continue

        local_t, _ = tune_threshold(probs[idx], labels[idx])
        # Shrink toward global threshold for stability on smaller modality slices.
        alpha = len(idx) / (len(idx) + shrink_k)
        blended = alpha * local_t + (1.0 - alpha) * global_threshold
        thresholds[modality] = float(blended)

    return thresholds


def apply_thresholds(
    probs: np.ndarray,
    modalities: np.ndarray,
    global_threshold: float,
    by_modality: dict[str, float] | None = None,
) -> np.ndarray:
    if not by_modality:
        return (probs >= global_threshold).astype(int)

    pred = np.zeros(len(probs), dtype=int)
    for i, prob in enumerate(probs):
        threshold = by_modality.get(modalities[i], global_threshold)
        pred[i] = 1 if prob >= threshold else 0
    return pred


def main() -> None:
    args = parse_args()

    payload = load_dataset(args.data)
    labeled = flatten_labeled_pairs(payload)
    if not labeled:
        raise RuntimeError("No labeled pairs found in dataset")

    records = [x.record for x in labeled]
    labels = np.asarray([x.label for x in labeled], dtype=int)
    groups = pick_groups(records)
    modalities = np.asarray(
        [extract_modality(normalize_description(r.current_description)) for r in records],
        dtype=object,
    )

    unique_groups = np.unique(groups)
    n_splits = min(args.cv_folds, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Need at least 2 unique patient/case groups for CV")

    configs = [
        {"c": 0.5, "class_weight": None},
        {"c": 1.0, "class_weight": None},
        {"c": 1.0, "class_weight": "balanced"},
    ]

    cv_scores: dict[str, list[float]] = {config_key(cfg): [] for cfg in configs}
    cv_thresholds: dict[str, list[float]] = {config_key(cfg): [] for cfg in configs}
    oof_probs: dict[str, np.ndarray] = {config_key(cfg): np.full(len(records), np.nan, dtype=np.float32) for cfg in configs}

    splitter = GroupKFold(n_splits=n_splits)

    for fold_num, (train_idx, val_idx) in enumerate(splitter.split(records, labels, groups), start=1):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        feature_artifacts, x_train = build_feature_matrix_fit(train_records)
        x_val = build_feature_matrix_transform(val_records, feature_artifacts)

        for cfg in configs:
            model = LogisticRegression(
                C=float(cfg["c"]),
                class_weight=cfg["class_weight"],
                max_iter=900,
                solver="saga",
                random_state=args.seed,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            probs = model.predict_proba(x_val)[:, 1]
            threshold, fold_acc = tune_threshold(probs, y_val)

            key = config_key(cfg)
            cv_scores[key].append(float(fold_acc))
            cv_thresholds[key].append(float(threshold))
            oof_probs[key][val_idx] = probs.astype(np.float32)

        print(f"Completed CV fold {fold_num}/{n_splits}", flush=True)

    best_cfg = None
    best_key = None
    best_oof_acc = -1.0
    best_oof_threshold = 0.5

    for cfg in configs:
        key = config_key(cfg)
        probs = oof_probs[key]
        if np.isnan(probs).any():
            raise RuntimeError(f"OOF predictions incomplete for {key}")

        threshold, oof_acc = tune_threshold(probs, labels)
        if oof_acc > best_oof_acc:
            best_oof_acc = oof_acc
            best_oof_threshold = threshold
            best_cfg = cfg
            best_key = key

    assert best_cfg is not None
    assert best_key is not None

    selected_modality_thresholds = tune_modality_thresholds(
        modalities=modalities,
        probs=oof_probs[best_key],
        labels=labels,
        global_threshold=best_oof_threshold,
        min_samples=args.mod_threshold_min_samples,
        shrink_k=args.mod_threshold_shrink_k,
    )

    # Refit final vectorizers and model on all available labeled data.
    feature_artifacts, x_all = build_feature_matrix_fit(records)
    final_model = LogisticRegression(
        C=float(best_cfg["c"]),
        class_weight=best_cfg["class_weight"],
        max_iter=900,
        solver="saga",
        random_state=args.seed,
        n_jobs=-1,
    )
    final_model.fit(x_all, labels)

    all_probs = final_model.predict_proba(x_all)[:, 1]
    all_pred_global = apply_thresholds(all_probs, modalities, best_oof_threshold)
    all_pred_mod = apply_thresholds(all_probs, modalities, best_oof_threshold, selected_modality_thresholds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_config = {
        "global": float(best_oof_threshold),
        "by_modality": selected_modality_thresholds,
        "min_samples": int(args.mod_threshold_min_samples),
        "shrink_k": float(args.mod_threshold_shrink_k),
    }

    artifact = {
        "model": final_model,
        "word_vectorizer": feature_artifacts.word_vectorizer,
        "char_vectorizer": feature_artifacts.char_vectorizer,
        "threshold": float(best_oof_threshold),
        "threshold_config": threshold_config,
        "meta": {
            "selected_config": best_cfg,
            "cv_folds": int(n_splits),
            "cv_mean_accuracy": float(np.mean(cv_scores[best_key])),
            "oof_accuracy": float(best_oof_acc),
            "pairs_total": int(len(records)),
            "seed": args.seed,
        },
    }
    joblib.dump(artifact, out_path)

    cv_table = []
    for cfg in configs:
        key = config_key(cfg)
        fold_scores = cv_scores[key]
        fold_thresholds = cv_thresholds[key]
        tuned_t, tuned_oof = tune_threshold(oof_probs[key], labels)
        cv_table.append(
            {
                "config": cfg,
                "mean_accuracy": float(np.mean(fold_scores)),
                "std_accuracy": float(np.std(fold_scores)),
                "mean_threshold": float(np.mean(fold_thresholds)),
                "oof_best_threshold": float(tuned_t),
                "oof_accuracy": float(tuned_oof),
                "fold_accuracies": [float(x) for x in fold_scores],
            }
        )

    cv_table.sort(key=lambda x: x["oof_accuracy"], reverse=True)

    metrics = {
        "selected_config": best_cfg,
        "threshold_global": float(best_oof_threshold),
        "threshold_by_modality": selected_modality_thresholds,
        "cv_folds": int(n_splits),
        "cv_mean_accuracy": float(np.mean(cv_scores[best_key])),
        "oof_accuracy": float(best_oof_acc),
        "full_train_accuracy_global_threshold": float(accuracy_score(labels, all_pred_global)),
        "full_train_accuracy_modality_thresholds": float(accuracy_score(labels, all_pred_mod)),
        "pairs_total": int(len(records)),
        "positive_rate": float(labels.mean()),
        "cv_results": cv_table,
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model artifact to {out_path}")


if __name__ == "__main__":
    main()
