from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import PredictRequest, PredictResponse, Prediction
from src.model.features import PairRecord
from src.model.infer import InferenceBundle, load_bundle, predict_pairs


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("relevant-priors-api")


app = FastAPI(title="Relevant Priors API", version="1.0.0")


@lru_cache(maxsize=1)
def get_bundle() -> InferenceBundle:
    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    return load_bundle(model_path)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id") or request.headers.get("x-amzn-trace-id") or "unknown"

    total_cases = len(payload.cases)
    total_priors = sum(len(case.prior_studies) for case in payload.cases)
    logger.info("request_id=%s cases=%s priors=%s", request_id, total_cases, total_priors)

    rows: list[PairRecord] = []
    index: list[tuple[str, str]] = []

    for case in payload.cases:
        curr = case.current_study
        for prior in case.prior_studies:
            rows.append(
                PairRecord(
                    case_id=str(case.case_id),
                    patient_id=str(case.patient_id or ""),
                    prior_study_id=str(prior.study_id),
                    current_description=str(curr.study_description),
                    current_date=str(curr.study_date),
                    prior_description=str(prior.study_description),
                    prior_date=str(prior.study_date),
                )
            )
            index.append((str(case.case_id), str(prior.study_id)))

    bundle = get_bundle()
    labels = predict_pairs(rows, bundle)

    if len(labels) != len(index):
        raise HTTPException(status_code=500, detail="Prediction count mismatch")

    predictions = [
        Prediction(case_id=case_id, study_id=study_id, predicted_is_relevant=bool(label))
        for (case_id, study_id), label in zip(index, labels)
    ]

    if len(predictions) != total_priors:
        raise HTTPException(status_code=500, detail="Missing predictions")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "request_id=%s done priors=%s elapsed_ms=%s",
        request_id,
        len(predictions),
        elapsed_ms,
    )

    return PredictResponse(predictions=predictions)
