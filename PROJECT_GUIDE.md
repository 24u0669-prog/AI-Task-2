# Sentiment Studio | Portfolio Project Guide

## Start the web app

Double-click `run_dashboard.bat`. It creates the Python 3.13 environment if
needed, installs project requirements, starts the local web server, and opens
`index.html` in your browser.

Alternatively, activate the environment and start it from a terminal:

```powershell
.\.venv-py313\Scripts\Activate.ps1
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. FastAPI's interactive API playground is at
<http://127.0.0.1:8000/docs>. The first pretrained-model prediction downloads
DistilBERT weights; an internet connection is required for that initial setup.

## Features

- Single review inference with the pretrained DistilBERT sentiment model or the
  locally trained PyTorch CNN.
- Plain-language result notes explain the overall tone estimate and the
  limits of the model score; predictions do not include phrase-level
  explanations.
- Batch inference from pasted reviews or `.txt`/`.csv` files, plus CSV export.
- Review character count, ready-to-try examples, confidence scores, and a
  low-confidence note.
- Training API with a progress log and options for epochs, batch size, learning
  rate, and training subset.
- Held-out accuracy, macro precision/recall/F1, confusion matrix, and learning
  curves, with a plain-language summary of the test-set results.
- Local-only server binding and no review-history persistence.

## Train and predict from a terminal

```powershell
python train.py --epochs 15 --batch-size 64
python predict.py "The film had great performances and an engaging story."
python -m unittest -v test_text_model.py
```

The full corpus downloads into `data/`. Model weights, training history, plots,
and metrics are written to `artifacts/`. These local datasets and generated
artifacts are excluded from version control by `.gitignore`; retrain to
reproduce them after cloning.

## API example

```powershell
$body = @{ review = "A clever and moving film."; model = "transformer" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/predict" `
  -Method Post -ContentType "application/json" -Body $body
```

Endpoints: `GET /api/health`, `GET /api/metrics`, `POST /api/predict`,
`POST /api/predict/batch`, `POST /api/train`, and
`GET /api/train/{job_id}`.

## Technical overview

| Area | Implementation |
| --- | --- |
| Pretrained AI | DistilBERT sentiment model from Hugging Face Transformers |
| Trainable model | PyTorch embedding + multi-kernel 1D text CNN |
| Dataset | NLTK Movie Reviews corpus; balanced, deterministic train/validation/test splits |
| API | FastAPI, Pydantic request validation, Uvicorn |
| Interface | Responsive HTML/CSS/JavaScript dashboard; browser `fetch` calls |
| Quality checks | Unit tests for tokenization, vocabulary, short-text model inputs, splitting, and model-score mapping |

## Evaluation and limitations

The saved CNN was trained for five epochs on the complete available training
split. Its current held-out test accuracy is about 63%; macro F1 is about 61%.
This is a useful educational baseline, not a production guarantee. The corpus is
small and domain-specific, and the learned CNN can overfit. In the current
200-review test set, the CNN recognizes negative reviews more reliably than
positive reviews; the dashboard reports the current class-level recall
difference alongside the overall metrics. DistilBERT uses
pretrained sentiment weights and has separate behavior from the project CNN;
the CNN's test metrics do not measure DistilBERT. Confidence is a model score,
not calibrated certainty. Sarcasm, nuanced or mixed reviews, and topics unlike
movie reviews may be misclassified. Predictions classify overall tone; they do
not identify the phrases that caused a result.

## Interview talking points

1. Explain why training, validation, and test sets are split separately, and
   why the test split is not used to select epochs.
2. Compare a dataset-trained text CNN with a pretrained transformer in terms
   of speed, data requirements, context, and evaluation.
3. Describe tokenization, vocabulary/unknown tokens, padding, masked pooling,
   softmax probabilities, and macro F1.
4. Discuss the train/serve flow: FastAPI validation, cached model loading,
   batched inference, checkpointing, and browser-to-API requests.
5. Identify next experiments: more labeled data, early stopping, class-wise
   error analysis, calibration, transformer fine-tuning, and automated API/UI
   tests.
