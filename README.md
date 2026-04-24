# Relevant Priors Challenge API

Local implementation of an HTTP API that predicts whether each prior study is relevant to the current study.

## Project Structure

- `src/api/app.py`: FastAPI app (`/predict`)
- `src/api/schemas.py`: request/response models
- `src/model/features.py`: normalization and structured feature engineering
- `src/model/pipeline.py`: TF-IDF + structured feature matrix builders
- `src/model/data.py`: dataset flattening and truth mapping
- `src/model/infer.py`: artifact loading and batched prediction
- `train.py`: model training + threshold tuning
- `eval_local.py`: local accuracy and contract check
- `artifacts/`: trained model and training metrics

## Setup

Python `3.10+` is supported (current pins are compatible with Python 3.10).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python3 train.py --data relevant_priors_public.json --out artifacts/model.joblib
```

Training now uses grouped cross-validation by patient (fallback: case) to select logistic hyperparameters.
Thresholds are selected from out-of-fold predictions (global + optional modality-specific thresholds with shrinkage), then the model refits on all labeled pairs.

## Local Evaluation

```bash
python3 eval_local.py --data relevant_priors_public.json --model artifacts/model.joblib
```

## Run API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API returns one prediction for each prior study with fields:

- `case_id`
- `study_id`
- `predicted_is_relevant`

## Notes

- Inference is batched per request to avoid per-prior overhead.
- The API enforces output completeness (`len(predictions) == total priors`).
- Logging includes request id, case count, prior count, and latency.
