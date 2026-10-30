"""
ArecaGuard - Deep Learning Training Pipeline
Trains a CNN / Transfer Learning model to classify arecanut diseases across 5 classes.
"""

import os
import json
import logging
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "val")
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, "disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_CLASSES = 5


def build_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """
    Constructs a transfer-learning architecture with MobileNetV2 backbone.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    # Fine-tuning: freeze initial base model layers
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax", name="predictions"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def main():
    logger.info("Initializing ArecaGuard Model Training Pipeline...")
    logger.info(f"Train directory: {TRAIN_DIR}")
    logger.info(f"Validation directory: {VAL_DIR}")

    # Check if dataset directories exist and have images
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        logger.error("Dataset directories not found. Please ensure dataset/train and dataset/val are present.")
        return

    # Data Generators with Augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    try:
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
        )

        val_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

        logger.info(f"Detected classes: {train_generator.class_indices}")
        with open(CLASS_INDICES_PATH, "w") as f:
            json.dump(train_generator.class_indices, f, indent=4)

        if train_generator.samples == 0:
            logger.warning("No training samples found in dataset/train subfolders. Generating baseline template model...")
            model = build_model(num_classes=NUM_CLASSES)
            model.save(OUTPUT_MODEL_PATH)
            logger.info(f"Template model saved to {OUTPUT_MODEL_PATH}")
            return

        model = build_model(num_classes=len(train_generator.class_indices))

        callbacks = [
            ModelCheckpoint(
                filepath=OUTPUT_MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
        )

        logger.info("Training complete.")
        model.save(OUTPUT_MODEL_PATH)
        logger.info(f"Model saved to: {OUTPUT_MODEL_PATH}")

    except Exception as e:
        logger.error(f"Error occurred during training: {e}", exc_info=True)


if __name__ == "__main__":
    main()
