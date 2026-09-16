import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

try:
    import tensorflow as tf
    TENSORFLOW_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - handled gracefully in runtime
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(exc)

MODEL_LOAD_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "crop_leaf_model.keras"
CLASS_METADATA_PATH = MODEL_PATH.with_suffix(".classes.json")
EVALUATION_DIR = BASE_DIR / "evaluation"
DATA_INFO_PATH = BASE_DIR / "data" / "disease_info.json"
DB_PATH = BASE_DIR / "data" / "predictions.db"
LOW_CONFIDENCE_THRESHOLD = 40.0

CLASS_NAMES = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Spot",
    "Rust",
    "Powdery Mildew",
    "Mosaic Virus",
    "Nutrient Deficiency",
]

DISEASE_GUIDE = {
    "Healthy": {
        "crop": "General",
        "description": "The leaf appears healthy and does not show active disease symptoms.",
        "symptoms": "No visible lesions, chlorosis, or deformation.",
        "prevention": "Maintain balanced irrigation, nutrition, and field scouting.",
        "management": "Continue routine monitoring and healthy crop maintenance.",
        "severity": "Low",
        "urgency": "Low",
    },
    "Early Blight": {
        "crop": "Tomato",
        "description": "Targeted lesions spreading in concentric rings often on lower leaves.",
        "symptoms": "Dark lesions with concentric rings, yellow halo, and leaf drop.",
        "prevention": "Crop rotation, resistant varieties, and avoiding wet foliage.",
        "management": "Remove infected leaves and use a labeled fungicide when needed.",
        "severity": "Moderate",
        "urgency": "Moderate",
    },
    "Late Blight": {
        "crop": "Tomato / Potato",
        "description": "Fast-spreading fungal disease that can devastate foliage under cool wet conditions.",
        "symptoms": "Water-soaked lesions, rapid spread, and leaf collapse.",
        "prevention": "Improve airflow and avoid excess moisture on leaves.",
        "management": "Act quickly with a registered fungicide and remove infected tissue.",
        "severity": "High",
        "urgency": "High",
    },
    "Leaf Spot": {
        "crop": "Various crops",
        "description": "Localized spots caused by fungal or bacterial stress on foliage.",
        "symptoms": "Round or irregular spots with margins and yellowing around them.",
        "prevention": "Avoid overhead watering and improve field sanitation.",
        "management": "Prune affected sections and apply crop-appropriate protection if recommended.",
        "severity": "Moderate",
        "urgency": "Moderate",
    },
    "Rust": {
        "crop": "Cereal / Legume",
        "description": "Rust diseases create pustules and can spread rapidly under humid conditions.",
        "symptoms": "Orange, brown, or yellow rust pustules on leaves and stems.",
        "prevention": "Use resistant varieties and avoid dense canopy conditions.",
        "management": "Remove infected debris and apply a disease-specific fungicide if recommended.",
        "severity": "Moderate",
        "urgency": "Moderate",
    },
    "Powdery Mildew": {
        "crop": "Grapes / Cucurbits / Peas",
        "description": "White powder-like growth appears on leaf surfaces and can reduce photosynthesis.",
        "symptoms": "Fine white fungal coating on leaves and stems.",
        "prevention": "Increase airflow and reduce dense leaf canopies.",
        "management": "Apply sulfur-based or labeled mildew control solutions if appropriate.",
        "severity": "Moderate",
        "urgency": "Moderate",
    },
    "Mosaic Virus": {
        "crop": "Vegetable crops",
        "description": "Viral infection causes mottled patterns and leaf distortion.",
        "symptoms": "Mottled yellow-green patches, distortion, and reduced vigor.",
        "prevention": "Control vectors and remove infected plants early.",
        "management": "Remove infected plants and prevent vector movement between crops.",
        "severity": "High",
        "urgency": "High",
    },
    "Nutrient Deficiency": {
        "crop": "General",
        "description": "Symptoms are caused by imbalanced nutrient availability rather than a pathogen.",
        "symptoms": "Chlorosis, patterning, stunted growth, and leaf discoloration.",
        "prevention": "Regular soil testing and balanced nutrient management.",
        "management": "Correct nutrient deficiencies with soil or foliar feeding based on testing results.",
        "severity": "Moderate",
        "urgency": "Moderate",
    },
}


def _safe_json_load(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    return default


def _active_class_names():
    metadata = _safe_json_load(CLASS_METADATA_PATH, {})
    class_names = metadata.get("class_names") if isinstance(metadata, dict) else None
    if isinstance(class_names, list) and class_names:
        return class_names
    return CLASS_NAMES


def _active_model_info():
    metadata = _safe_json_load(CLASS_METADATA_PATH, {})
    model = app_model_cache if "app_model_cache" in globals() else None
    if isinstance(metadata, dict):
        return {
            "available": model is not None,
            "load_error": (TENSORFLOW_IMPORT_ERROR or MODEL_LOAD_ERROR) if model is None else None,
            "model": "MobileNet-based CNN-LSTM",
            "framework": "TensorFlow / Keras",
            "input_shape": list(model.input_shape[1:]) if model is not None else metadata.get("input_shape", [224, 224, 3]),
            "output_shape": list(model.output_shape[1:]) if model is not None else [len(metadata.get("class_names", _active_class_names()))],
            "dataset": metadata.get("dataset", "Synthetic demo dataset"),
            "classes": metadata.get("class_names", _active_class_names()),
        }
    return {
        "available": model is not None,
        "load_error": (TENSORFLOW_IMPORT_ERROR or MODEL_LOAD_ERROR) if model is None else None,
        "model": "MobileNet-based CNN-LSTM",
        "framework": "TensorFlow / Keras",
        "input_shape": [224, 224, 3],
        "dataset": "Synthetic demo dataset",
        "classes": _active_class_names(),
    }


def _ensure_directories():
    (UPLOAD_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(filename: str) -> str:
    safe_name = Path(filename).name
    safe_name = safe_name.replace(" ", "_")
    return "".join(ch for ch in safe_name if ch.isalnum() or ch in {"_", ".", "-"})


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    _ensure_directories()
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uploaded_name TEXT,
                saved_image_path TEXT,
                predicted_label TEXT,
                confidence REAL,
                crop TEXT,
                severity TEXT,
                urgency TEXT,
                disease_info TEXT,
                top_predictions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        migrations = {
            "saved_image_path": "ALTER TABLE predictions ADD COLUMN saved_image_path TEXT",
            "crop": "ALTER TABLE predictions ADD COLUMN crop TEXT",
            "severity": "ALTER TABLE predictions ADD COLUMN severity TEXT",
            "urgency": "ALTER TABLE predictions ADD COLUMN urgency TEXT",
            "disease_info": "ALTER TABLE predictions ADD COLUMN disease_info TEXT",
            "top_predictions": "ALTER TABLE predictions ADD COLUMN top_predictions TEXT",
        }
        for column, statement in migrations.items():
            if column not in columns:
                conn.execute(statement)
        conn.commit()


def _generate_demo_leaf_image(class_index: int, noise_seed: int):
    rng = np.random.default_rng(noise_seed)
    image = np.zeros((224, 224, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:224, 0:224]
    center_x = 112 + rng.normal(0, 8)
    center_y = 112 + rng.normal(0, 8)
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    leaf_mask = radius < 78 + rng.normal(0, 5)
    base_green = np.array([20 + rng.integers(30), 110 + rng.integers(30), 45 + rng.integers(20)], dtype=np.float32)
    image[leaf_mask] = base_green

    if class_index == 0:  # Healthy
        image[leaf_mask] += np.array([8, 20, 10], dtype=np.float32)
    elif class_index == 1:  # Early Blight
        blotches = (((xx - 80 - noise_seed) ** 2 + (yy - 120) ** 2) < 50) | (((xx - 150) ** 2 + (yy - 90) ** 2) < 36)
        image[blotches] = np.array([90, 65, 35], dtype=np.float32)
    elif class_index == 2:  # Late Blight
        wet = ((yy % 18 < 5) & (radius < 90)) | (((xx - 110) ** 2 + (yy - 120) ** 2) < 42)
        image[wet] = np.array([55, 85, 55], dtype=np.float32)
    elif class_index == 3:  # Leaf Spot
        spots = (((xx - 60) ** 2 + (yy - 140) ** 2) < 18) | (((xx - 140) ** 2 + (yy - 80) ** 2) < 16)
        image[spots] = np.array([105, 60, 35], dtype=np.float32)
    elif class_index == 4:  # Rust
        pustules = ((xx + noise_seed) % 18 < 4) & (yy % 20 < 5) & (radius < 90)
        image[pustules] = np.array([195, 120, 30], dtype=np.float32)
    elif class_index == 5:  # Powdery Mildew
        mildew = ((xx + yy + noise_seed) % 12 < 3) & (radius < 95)
        image[mildew] = np.array([240, 240, 230], dtype=np.float32)
    elif class_index == 6:  # Mosaic Virus
        yellow_lines = ((yy + noise_seed) % 14 < 4) | ((xx + noise_seed) % 15 < 3)
        image[yellow_lines & (radius < 94)] = np.array([210, 210, 70], dtype=np.float32)
    elif class_index == 7:  # Nutrient Deficiency
        zones = (((xx - 90) ** 2 + (yy - 120) ** 2) < 1200) | (((xx - 150) ** 2 + (yy - 85) ** 2) < 850)
        image[zones] = np.array([220, 200, 70], dtype=np.float32)

    vein_mask = (np.abs(xx - center_x) < 3) | (np.abs(yy - center_y) < 3)
    image[vein_mask] = np.array([18, 60, 35], dtype=np.float32)
    return image.astype(np.float32) / 255.0


def _build_demo_model(class_names):
    if tf is None:
        return None

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
        alpha=1.0,
    )
    inputs = tf.keras.Input(shape=(224, 224, 3))
    prepared = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    features = backbone(prepared, training=False)
    sequence = tf.keras.layers.Reshape((-1, features.shape[-1]))(features)
    x = tf.keras.layers.LSTM(128, return_sequences=False)(sequence)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    output = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _generate_demo_dataset(class_names):
    x, y = [], []
    for class_index, _ in enumerate(class_names):
        for sample_index in range(160):
            x.append(_generate_demo_leaf_image(class_index, class_index * 37 + sample_index))
            y.append(class_index)

    x = np.stack(x).astype(np.float32)
    y = np.asarray(y, dtype=np.int32)
    indices = np.arange(len(y))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]
    split_train = int(0.7 * len(y))
    split_val = int(0.15 * len(y))
    train_x, train_y = x[:split_train], y[:split_train]
    val_x, val_y = x[split_train:split_train + split_val], y[split_train:split_train + split_val]
    test_x, test_y = x[split_train + split_val:], y[split_train + split_val:]
    return train_x, train_y, val_x, val_y, test_x, test_y


def _write_class_metadata(class_names, dataset_name="Synthetic demo dataset"):
    payload = {
        "class_names": class_names,
        "input_shape": [224, 224, 3],
        "dataset": dataset_name,
        "notes": "This project stores a real Keras model and class list; the dataset is a synthetic demo fallback because no public PlantVillage folder is present in the workspace.",
    }
    CLASS_METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_evaluation_metrics(class_names, true_labels, predicted_labels):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
        "precision": round(float(precision_score(true_labels, predicted_labels, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)), 4),
    }
    (EVALUATION_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_text = classification_report(true_labels, predicted_labels, target_names=class_names)
    (EVALUATION_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    tick_positions = np.arange(len(class_names))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def _restore_or_build_model():
    global MODEL_LOAD_ERROR
    if tf is None:
        return None
    class_names = _active_class_names()
    if not MODEL_PATH.exists() or not CLASS_METADATA_PATH.exists():
        MODEL_LOAD_ERROR = "Saved model or class metadata is missing."
        return None
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
    except Exception as exc:  # pragma: no cover - depends on local Keras runtime
        MODEL_LOAD_ERROR = str(exc)
        print(f"Model load failed: {MODEL_LOAD_ERROR}")
        return None
    if model is None or model.output_shape[-1] != len(class_names):
        MODEL_LOAD_ERROR = "Saved model output classes do not match the class metadata."
        return None
    print("===== MODEL DEBUG =====")
    print(f"Model: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    return model


def _load_model():
    if tf is None:
        return None
    return _restore_or_build_model()


def _preprocess_input_image(image: Image.Image, model=None):
    rgb_image = image.convert("RGB")
    model_shape = model.input_shape if model is not None else (None, 224, 224, 3)
    height, width = model_shape[1:3]
    resized = rgb_image.resize((width, height))
    array = np.asarray(resized, dtype=np.float32)
    metadata = _safe_json_load(CLASS_METADATA_PATH, {})
    if metadata.get("dataset") == "Synthetic demo dataset" and "preprocessing" not in metadata:
        array /= 255.0
    array = np.expand_dims(array, 0)
    return array


def _probability_vector_from_model(model, image):
    if model is None:
        raise RuntimeError("The trained disease model is unavailable in this Python environment.")
    batch = _preprocess_input_image(image, model)
    probabilities = model.predict(batch, verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.size == 0:
        raise RuntimeError("The trained disease model returned an empty probability vector.")
    probabilities = probabilities / np.sum(probabilities + 1e-8)
    return probabilities


def _classify_image(image: Image.Image):
    model = app_model_cache if app_model_cache is not None else _load_model()
    class_names = _active_class_names()
    if model is None:
        detail = TENSORFLOW_IMPORT_ERROR or MODEL_LOAD_ERROR or "The saved model could not be loaded."
        raise RuntimeError(f"The trained disease model is unavailable: {detail}")
    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output classes do not match the class metadata.")
    probabilities = _probability_vector_from_model(model, image)
    index = int(np.argmax(probabilities))
    confidence = float(np.clip(probabilities[index] * 100.0, 0.0, 100.0))
    print("========== PREDICTION DEBUG ==========")
    print(f"Image: {image.size[0]}x{image.size[1]}")
    print("Processed image: (224, 224, 3)")
    print(f"Model input: {model.input_shape}")
    print(f"Model output: {model.output_shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Probabilities: {[round(float(value) * 100, 4) for value in probabilities]}")
    print(f"Predicted index: {index}")
    print(f"Predicted class: {class_names[index]}")
    print(f"Confidence: {confidence:.2f}")
    print("=======================================")
    return class_names[index], round(confidence, 2), probabilities.tolist()


def _safe_open_image(file_data: bytes):
    try:
        return Image.open(io.BytesIO(file_data)).convert("RGB")
    except (UnidentifiedImageError, ValueError, OSError, SyntaxError):
        raise ValueError("Uploaded file is not a valid image.")


def _save_uploaded_image(file_name: str, image: Image.Image):
    _ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = _sanitize_filename(file_name or "leaf_image.png")
    base_name = safe_name.rsplit(".", 1)[0] or "leaf_image"
    target_name = f"{timestamp}_{base_name}.jpg"
    target_path = UPLOAD_DIR / "predictions" / target_name
    image.convert("RGB").save(target_path, format="JPEG", quality=90)
    return str(target_path.relative_to(BASE_DIR)).replace("\\", "/")


def _get_disease_details(label: str):
    details = DISEASE_GUIDE.get(label, {
        "crop": "General",
        "description": "This disease is not in the reference list. Consult an agronomist for confirmation.",
        "symptoms": "General plant stress or unknown disease pattern.",
        "prevention": "Improve crop monitoring and field inspection.",
        "management": "Consult a local agronomy specialist.",
        "severity": "Unknown",
        "urgency": "Unknown",
    })
    return {
        "label": label,
        "crop": details.get("crop", "General"),
        "description": details.get("description", "No description available."),
        "symptoms": details.get("symptoms", "See a crop specialist."),
        "prevention": details.get("prevention", "No prevention guidance available."),
        "management": details.get("management", "No management guidance available."),
        "severity": details.get("severity", "Unknown"),
        "urgency": details.get("urgency", "Unknown"),
    }


def _top_predictions(probabilities, class_names):
    ranked = sorted(
        enumerate(probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return [
        {
            "label": class_names[index],
            "probability": round(float(probability) * 100.0, 2),
        }
        for index, probability in ranked
    ]


def _log_prediction(uploaded_name: str, saved_image_path: str, label: str, confidence: float, crop: str, severity: str, urgency: str, disease_info: dict, top_predictions: list):
    _init_db()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                uploaded_name, saved_image_path, predicted_label, confidence, crop, severity, urgency, disease_info, top_predictions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uploaded_name,
                saved_image_path,
                label,
                confidence,
                crop,
                severity,
                urgency,
                json.dumps(disease_info),
                json.dumps(top_predictions),
            ),
        )

        conn.execute(
            """
            DELETE FROM predictions
            WHERE id NOT IN (
                SELECT id FROM predictions
                ORDER BY id DESC
                LIMIT 5
            )
            """
        )
        conn.commit()


def create_app(testing: bool = False):
    app = Flask(__name__)
    app.config["TESTING"] = testing

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/diseases")
    def diseases():
        return render_template("diseases.html", diseases=_active_class_names(), guide=DISEASE_GUIDE)

    @app.route("/diseases/<path:disease_name>")
    def disease_information(disease_name):
        label = disease_name.replace("-", " ")
        matching_label = next((name for name in _active_class_names() if name.lower() == label.lower()), label)
        disease = _get_disease_details(matching_label)
        return render_template("disease.html", disease=matching_label, details=disease)

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(UPLOAD_DIR, filename)

    @app.route("/predict", methods=["POST"])
    def predict_legacy():
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No image file uploaded."}), 400
        try:
            file_data = file.read()
            if not file_data:
                return jsonify({"error": "Uploaded image is empty."}), 400
            image = _safe_open_image(file_data)
            label, confidence, probabilities = _classify_image(image)
            disease_info = _get_disease_details(label)
            top_predictions = _top_predictions(probabilities, _active_class_names())
            saved_name = file.filename or "upload.png"
            saved_image_path = _save_uploaded_image(saved_name, image)
            _log_prediction(saved_name, saved_image_path, label, confidence, disease_info["crop"], disease_info["severity"], disease_info["urgency"], disease_info, top_predictions)
            return jsonify({
                "label": label,
                "prediction": label,
                "confidence": confidence,
                "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
                "confidence_message": "The model could not confidently identify this leaf." if confidence < LOW_CONFIDENCE_THRESHOLD else None,
                "status": "success",
                "treatment": disease_info.get("management", "Consult an agronomist for guidance."),
                "disease_info": disease_info,
                "top_predictions": top_predictions,
                "history_path": f"/uploads/{saved_image_path.replace('uploads/', '')}",
                "success": True,
            })
        except ValueError as exc:
            return jsonify({"error": str(exc), "success": False}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc), "success": False, "code": "model_unavailable"}), 503
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return jsonify({"error": f"Unable to process image: {exc}", "success": False}), 400

    @app.route("/api/health")
    def api_health():
        return jsonify({
            "status": "ok",
            "message": "Crop disease service is online.",
            "model_available": app_model_cache is not None,
            "classes": _active_class_names(),
        })

    @app.route("/api/diseases")
    def api_diseases():
        return jsonify([
            {
                "label": disease,
                "crop": details.get("crop", "General"),
                "description": details.get("description", "No description available."),
                "severity": details.get("severity", "Unknown"),
                "urgency": details.get("urgency", "Unknown"),
            }
            for disease, details in DISEASE_GUIDE.items()
        ])

    @app.route("/api/model-info")
    def api_model_info():
        return jsonify(_active_model_info())

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        return predict_legacy()

    @app.route("/api/history")
    def api_history():
        with _get_connection() as conn:
            records = conn.execute(
                "SELECT id, uploaded_name, saved_image_path, predicted_label, confidence, crop, severity, urgency, disease_info, top_predictions, created_at FROM predictions ORDER BY id DESC"
            ).fetchall()
        payload = []
        for record in records:
            disease_info = json.loads(record["disease_info"] or "{}")
            top_predictions = json.loads(record["top_predictions"] or "[]")
            payload.append({
                "id": record["id"],
                "uploaded_name": record["uploaded_name"],
                "prediction": record["predicted_label"],
                "confidence": record["confidence"],
                "crop": record["crop"],
                "severity": record["severity"],
                "urgency": record["urgency"],
                "disease_info": disease_info,
                "top_predictions": top_predictions,
                "created_at": record["created_at"],
                "image_url": f"/uploads/{record['saved_image_path'].replace('uploads/', '')}" if record["saved_image_path"] else None,
            })
        return jsonify(payload)

    @app.route("/api/history/<int:history_id>")
    def api_history_detail(history_id):
        with _get_connection() as conn:
            record = conn.execute(
                "SELECT id, uploaded_name, saved_image_path, predicted_label, confidence, crop, severity, urgency, disease_info, top_predictions, created_at FROM predictions WHERE id = ?",
                (history_id,),
            ).fetchone()
        if not record:
            return jsonify({"error": "History entry not found."}), 404
        disease_info = json.loads(record["disease_info"] or "{}")
        top_predictions = json.loads(record["top_predictions"] or "[]")
        return jsonify({
            "id": record["id"],
            "uploaded_name": record["uploaded_name"],
            "prediction": record["predicted_label"],
            "confidence": record["confidence"],
            "crop": record["crop"],
            "severity": record["severity"],
            "urgency": record["urgency"],
            "disease_info": disease_info,
            "top_predictions": top_predictions,
            "created_at": record["created_at"],
            "image_url": f"/uploads/{record['saved_image_path'].replace('uploads/', '')}" if record["saved_image_path"] else None,
        })

    @app.route("/api/history/<int:history_id>", methods=["DELETE"])
    def api_history_delete(history_id):
        with _get_connection() as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (history_id,))
            conn.commit()
        return jsonify({"success": True, "deleted": cursor.rowcount > 0})

    @app.route("/api/stats")
    def api_stats():
        with _get_connection() as conn:
            total_scans = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            average_confidence = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
            most_common = conn.execute(
                "SELECT predicted_label, COUNT(*) AS count FROM predictions GROUP BY predicted_label ORDER BY count DESC LIMIT 1"
            ).fetchone()
            recent = conn.execute(
                "SELECT predicted_label, confidence, created_at FROM predictions ORDER BY id DESC LIMIT 5"
            ).fetchall()
        return jsonify({
            "total_scans": total_scans,
            "average_confidence": round(float(average_confidence or 0.0), 2),
            "most_common_disease": dict(most_common) if most_common else None,
            "recent_activity": [
                {
                    "prediction": row["predicted_label"],
                    "confidence": row["confidence"],
                    "created_at": row["created_at"],
                }
                for row in recent
            ],
        })

    @app.route("/api/export")
    def api_export():
        with _get_connection() as conn:
            records = conn.execute(
                "SELECT id, uploaded_name, saved_image_path, predicted_label, confidence, crop, severity, urgency, disease_info, top_predictions, created_at FROM predictions ORDER BY id DESC"
            ).fetchall()
        export = []
        for record in records:
            export.append({
                "id": record["id"],
                "uploaded_name": record["uploaded_name"],
                "saved_image_path": record["saved_image_path"],
                "prediction": record["predicted_label"],
                "confidence": record["confidence"],
                "crop": record["crop"],
                "severity": record["severity"],
                "urgency": record["urgency"],
                "disease_info": json.loads(record["disease_info"] or "{}"),
                "top_predictions": json.loads(record["top_predictions"] or "[]"),
                "created_at": record["created_at"],
            })
        return jsonify({"records": export})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "classes": _active_class_names()})

    @app.route("/history")
    def history_legacy():
        return api_history()

    _init_db()
    return app


app_model_cache = None


def _bootstrap_model_once():
    global app_model_cache
    app_model_cache = _load_model()


_bootstrap_model_once()


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
