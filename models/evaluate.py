"""
ArecaGuard - Model Evaluation & Metrics Verification
Computes accuracy, precision, recall, F1-score, and confusion matrix on the validation set.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
VAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "val")
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
REPORT_OUTPUT = os.path.join(BASE_DIR, "evaluation_report.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Trained model not found at {MODEL_PATH}. Train the model first.")
        return

    logger.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    if val_generator.samples == 0:
        logger.warning("No validation samples found in dataset/val subdirectories. Add test images to evaluate.")
        return

    # Predictions
    logger.info("Computing predictions on validation dataset...")
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    labels = list(val_generator.class_indices.keys())

    report_text = classification_report(y_true, y_pred, target_names=labels)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(report_text)
    print("\nCONFUSION MATRIX:")
    print(cm)
    print("=" * 50)

    # Save summary report
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_dict["confusion_matrix"] = cm.tolist()

    with open(REPORT_OUTPUT, "w") as f:
        json.dump(report_dict, f, indent=4)

    logger.info(f"Evaluation report saved to: {REPORT_OUTPUT}")


if __name__ == "__main__":
    evaluate()
