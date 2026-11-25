import json
import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.utils.grad_cam import generate_gradcam, get_img_array
from models.train_model import build_model

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "disease_model.h5")
CLASS_INDICES_PATH = os.path.join(PROJECT_ROOT, "models", "class_indices.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_CLASSES = [
    "Healthy",
    "Mahali / Koleroga Detected",
    "Anthracnose Detected",
    "Fruit Split Detected",
    "Bacterial Blotch Detected",
]

ADVICE_MAP = {
    "Healthy": "No disease symptoms detected. Maintain routine orchard hygiene, balanced nutrition, and regular field monitoring.",
    "Mahali / Koleroga Detected": "Spray 1% Bordeaux mixture on the infected nuts and surrounding palms immediately. Ensure proper drainage in the field to prevent further fungal spread.",
    "Anthracnose Detected": "Prune and destroy severely infected parts, improve canopy airflow, and apply recommended copper-based or systemic fungicides as advised.",
    "Fruit Split Detected": "Maintain uniform soil moisture and proper irrigation schedules to reduce sudden water stress and nutrient imbalances.",
    "Bacterial Blotch Detected": "Remove affected fruits/tissues, avoid overhead irrigation where possible, and follow recommended bactericide/sanitation practices.",
}


def load_class_names():
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            indices = json.load(f)
        if indices:
            return [k for k, _ in sorted(indices.items(), key=lambda item: item[1])]
    return DEFAULT_CLASSES


def load_areca_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024:
        return tf.keras.models.load_model(MODEL_PATH)
    return build_model(num_classes=len(load_class_names()))


model = load_areca_model()
class_names = load_class_names()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/result", methods=["POST"])
def predict():
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please select an image to upload.")

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    img_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(img_path)

    img_array = get_img_array(img_path, size=(224, 224))
    preds = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence_score = round(float(preds[0][pred_idx]) * 100, 2)
    disease_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
    advice_en = ADVICE_MAP.get(
        disease_name,
        "Consult a local agricultural extension officer and inspect nearby palms for similar symptoms.",
    )

    heatmap_filename = f"gradcam_{unique_name}"
    heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
    generate_gradcam(
        img_path=img_path,
        model=model,
        last_conv_layer_name="Conv_1",
        output_path=heatmap_path,
    )

    return render_template(
        "result.html",
        disease_name=disease_name,
        confidence_score=confidence_score,
        advice_en=advice_en,
        image_path=url_for("static", filename=f"uploads/{unique_name}"),
        heatmap_path=url_for("static", filename=f"uploads/{heatmap_filename}"),
    )


if __name__ == "__main__":
    app.run(debug=True)