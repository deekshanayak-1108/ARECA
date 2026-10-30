import os
import cv2
import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DEFAULT_IMAGE_SIZE = (224, 224)

# Canonical class labels matching directory names
CLASS_NAMES = [
    "anthracnose",
    "bacterial_blotch",
    "fruit_split",
    "healthy",
    "mahali_koleroga",
]

# Display metadata and advisory recommendations
DISEASE_METADATA = {
    "healthy": {
        "title": "Healthy Arecanut",
        "scientific_name": "Areca catechu (Healthy)",
        "severity": "None",
        "description": "The arecanut fruit appears vigorous, unblemished, and free of visible fungal or bacterial lesions.",
        "management": [
            "Maintain regular orchard sanitation and weed clearance.",
            "Continue scheduled balanced NPK fertilization and organic compost application.",
            "Inspect bunch stalks periodically, especially prior to pre-monsoon showers.",
        ],
    },
    "mahali_koleroga": {
        "title": "Mahali / Koleroga (Fruit Rot)",
        "scientific_name": "Phytophthora meadii",
        "severity": "Critical",
        "description": "Destructive water-soaked lesions leading to rotting and massive premature fruit drop during continuous heavy monsoons.",
        "management": [
            "Prophylactic Spray: Apply 1% Bordeaux mixture before monsoon onset (late May / early June).",
            "Follow-up Spray: Repeat Bordeaux mixture spray after 40–45 days during monsoon breaks.",
            "Curative Action: Apply Metalaxyl-Mancozeb (2.5 g/L) or Fosetyl-Al (2 g/L) if disease has already appeared.",
            "Tie polythene covers over arecanut bunches in endemic, high-rainfall regions.",
            "Collect and burn fallen infected nuts to prevent soil fungal inoculum build-up.",
        ],
    },
    "anthracnose": {
        "title": "Anthracnose (Die-back / Inflorescence Rot)",
        "scientific_name": "Colletotrichum gloeosporioides",
        "severity": "High",
        "description": "Dark brown sunken circular lesions on green tender nuts, accompanied by drying and yellowing of surrounding tissues.",
        "management": [
            "Spray Mancozeb (2 g/L) or Carbendazim (1 g/L) at 15-day intervals upon noticing early spots.",
            "Ensure proper drainage in the plantation to minimize canopy humidity.",
            "Prune infected branches and inflorescences and safely dispose of debris.",
        ],
    },
    "fruit_split": {
        "title": "Fruit Split (Physiological Cracking)",
        "scientific_name": "Physiological / Nutritional Disorder",
        "severity": "Moderate",
        "description": "Cracking near the calyx or vertical splitting of ripening nuts, often caused by drought followed by sudden heavy irrigation/rain.",
        "management": [
            "Maintain consistent soil moisture levels; avoid sudden flood irrigation after dry spells.",
            "Foliar spray of Borax (2 g/L) and Potassium Sulphate (5 g/L) during kernel filling stages.",
            "Mulch the base of palm basins to regulate moisture retention and root temperature.",
        ],
    },
    "bacterial_blotch": {
        "title": "Bacterial Blotch",
        "scientific_name": "Xanthomonas / Pseudomonas spp.",
        "severity": "High",
        "description": "Water-soaked dark brownish to black blotches with oily margins on tender and mature nuts.",
        "management": [
            "Spray Copper Oxychloride (2.5 g/L) combined with Streptomycin Sulphate (0.5 g/L).",
            "Avoid overhead sprinkler irrigation that splashes bacteria between palms.",
            "Disinfect harvesting tools between trees to prevent mechanical transmission.",
        ],
    },
}


def allowed_file(filename: str) -> bool:
    """Checks whether the file has a supported image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: str, target_size=DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """
    Loads and standardizes an image for model input:
    - Resizes to target dimensions
    - Converts BGR to RGB
    - Normalizes pixel values to [0, 1]
    - Adds batch dimension: shape (1, H, W, C)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image via OpenCV
    img = cv2.imread(image_path)
    if img is None:
        # Fallback to PIL in case of format quirks
        pil_img = Image.open(image_path).convert("RGB")
        img = np.array(pil_img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_array = img.astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def get_disease_info(class_name: str) -> dict:
    """Retrieves metadata and mitigation recommendations for a predicted class."""
    return DISEASE_METADATA.get(
        class_name,
        {
            "title": class_name.replace("_", " ").title(),
            "scientific_name": "N/A",
            "severity": "Unknown",
            "description": "No detailed metadata registered for this condition.",
            "management": ["Consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer."],
        },
    )
