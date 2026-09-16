# Crop Leaf Disease Identifier

This project is a Flask-based crop leaf disease detector with a MobileNet-inspired CNN-LSTM-style pipeline, uploaded-image processing, SQLite logging, and a lightweight web UI.

## Overview

The app accepts a crop leaf image, validates it, preprocesses it, runs the model, and returns a disease label with confidence. Each prediction is saved in SQLite so the app can show history and aggregated statistics.

## Features

- Image upload and basic validation
- MobileNet-based feature extraction with an LSTM classification head
- Flask API with prediction, health, history, and statistics routes
- SQLite persistence for prediction records
- Disease reference pages for supported classes
- Evaluation output generated from the model on synthetic demo data
- Responsive dashboard-style frontend

## Architecture

- `app.py` – Flask backend, route definitions, preprocessing, model loading, and persistence
- `templates/` – frontend pages
- `static/style.css` – styling and responsive layout
- `models/` – Keras model and class metadata
- `data/` – SQLite database and disease metadata
- `evaluation/` – metrics and confusion matrix artifacts
- `tests/` – regression tests for app behaviour

## Technology stack

- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- scikit-learn
- SQLite
- Matplotlib

## Model architecture

The model uses a MobileNetV2-style backbone followed by reshaping into a sequence and an LSTM layer. This creates a valid CNN-LSTM architecture for feature extraction and sequence-level reasoning before classification into the supported crop disease classes.

## Preprocessing

Before prediction, the app:

- validates the file and image format
- converts the uploaded image to RGB
- resizes to 224x224
- normalizes pixel values to the range expected by the model
- passes the sample through the single cached model instance

## Dataset

The workspace does not include a public PlantVillage directory, so the current persistence uses a synthetic demo dataset to keep the application runnable and testable while remaining honest about the actual data source.

## Evaluation

The project generates evaluation artifacts in the `evaluation/` directory, including:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`

These values are produced using scikit-learn metrics from the model’s actual test split outputs.

## Installation

```bash
python -m venv .venv
. .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Running the app

```bash
.venv\Scripts\python.exe app.py
```

Then visit:

- http://127.0.0.1:5000

## API endpoints

- `GET /api/health`
- `POST /api/predict`
- `GET /api/history`
- `GET /api/history/<id>`
- `DELETE /api/history/<id>`
- `GET /api/stats`
- `GET /api/diseases`
- `GET /api/model-info`
- `GET /api/export`

## Database

Predictions are stored in SQLite at `data/predictions.db`.

## Testing

```bash
.venv\Scripts\python.exe -m pytest -q
```

## Known limitations

- No real PlantVillage dataset is present in this workspace.
- The saved model is a MobileNetV2 feature extractor plus LSTM classifier, but it was trained on a synthetic demo dataset rather than a full public plant disease corpus; its recorded evaluation is near chance (accuracy 0.1302), so it must not be treated as a reliable field diagnostic model.
- The application refuses to fabricate a prediction when TensorFlow is unavailable. Use the workspace `.venv` commands above; the system Python installation may not include TensorFlow.
- Camera support is browser-based and depends on browser permissions.

## Future improvements

- Add a real public dataset such as PlantVillage
- Extend the frontend to richer disease charts and modules
- Add stronger image quality checks and confidence thresholds
- Add a dedicated model training script for larger datasets
