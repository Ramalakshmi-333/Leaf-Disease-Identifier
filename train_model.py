"""Train the crop classifier on a PlantVillage-style directory dataset."""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from app import MODEL_PATH


IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train a crop disease image classifier.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing one subdirectory per class.")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Output Keras model path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_transfer_model(class_count: int):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    augmented = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ])(inputs)
    features = backbone(tf.keras.applications.mobilenet_v2.preprocess_input(augmented), training=False)
    sequence = tf.keras.layers.Reshape((-1, features.shape[-1]))(features)
    recurrent = tf.keras.layers.LSTM(128)(sequence)
    dense = tf.keras.layers.Dense(64, activation="relu")(recurrent)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(tf.keras.layers.Dropout(0.2)(dense))
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_from_directory(data_dir: Path, output_path: Path, epochs: int, batch_size: int, validation_split: float, seed: int):
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=IMAGE_SIZE, batch_size=batch_size,
        validation_split=validation_split, subset="training", seed=seed, label_mode="int"
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=IMAGE_SIZE, batch_size=batch_size,
        validation_split=validation_split, subset="validation", seed=seed, label_mode="int"
    )
    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise ValueError("The dataset must contain at least two class directories.")
    model = build_transfer_model(len(class_names))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True),
    ]
    history = model.fit(
        train_ds.prefetch(AUTOTUNE), validation_data=validation_ds.prefetch(AUTOTUNE),
        epochs=epochs, callbacks=callbacks
    )
    model.save(output_path)
    metadata_path = output_path.with_suffix(".classes.json")
    metadata_path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "input_shape": [*IMAGE_SIZE, 3],
                "image_size": IMAGE_SIZE,
                "dataset": str(data_dir),
                "preprocessing": "MobileNetV2 preprocess_input inside model",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = {key: round(float(values[-1]), 4) for key, values in history.history.items() if values}
    print(f"Model saved to: {output_path}")
    print(f"Class metadata saved to: {metadata_path}")
    print(f"Classes: {class_names}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    args = parse_args()
    if args.data_dir:
        train_from_directory(args.data_dir, args.output, args.epochs, args.batch_size, args.validation_split, args.seed)
    else:
        raise ValueError("A labeled dataset directory is required. Refusing to train a fake demo model.")
