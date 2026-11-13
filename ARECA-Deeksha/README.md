# ArecaGuard: AI-Powered Arecanut Disease Detection & Advisory Platform

**ArecaGuard** is an intelligent computer vision and deep learning web platform designed to assist arecanut farmers, agricultural officers, and researchers in identifying and managing crop diseases at early stages. The platform leverages Convolutional Neural Networks (CNNs) and Explainable AI (Grad-CAM) to provide transparent, accurate diagnoses and actionable mitigation strategies.

---

## 👥 Project Information

- **Institution**: Sahyadri College of Engineering & Management, Mangaluru
- **Department**: Computer Science & Engineering / Information Science & Engineering
- **Project Title**: ArecaGuard – Automated Arecanut Disease Classification and Diagnosis System

### 🎓 Team Members
1. **Deeksha K**
2. **Likitha GP**
3. **Shivani K S**
4. **Shekhar Vinayak**

### 👨‍🏫 Project Guide
- **Mr. Ganaraj K**, Assistant Professor, Sahyadri College of Engineering & Management

---

## 🎯 Target Diseases & Classes

The system is trained to identify the following 5 critical classes of arecanut health conditions:

1. **Healthy (`healthy`)**: Free of fungal/bacterial infections; normal fruit development.
2. **Mahali / Koleroga (`mahali_koleroga`)**: Fruit rot caused by *Phytophthora meadii*, notorious for causing massive premature nut drop during monsoon.
3. **Anthracnose (`anthracnose`)**: Fungal infection caused by *Colletotrichum gloeosporioides*, causing dark brown necrotic spots on nuts and inflorescence.
4. **Fruit Split (`fruit_split`)**: Cracking and premature splitting of nuts caused by physiological stress, water fluctuations, and micronutrient imbalances.
5. **Bacterial Blotch (`bacterial_blotch`)**: Bacterial dark blotching leading to fruit decay and reduced kernel quality.

---

## 📁 Repository Structure

```text
arecanut-guard/
├── dataset/
│   ├── train/
│   │   ├── healthy/
│   │   ├── mahali_koleroga/
│   │   ├── anthracnose/
│   │   ├── fruit_split/
│   │   └── bacterial_blotch/
│   └── val/
│       ├── healthy/
│       ├── mahali_koleroga/
│       ├── anthracnose/
│       ├── fruit_split/
│       └── bacterial_blotch/
├── models/
│   ├── train_model.py          # Model training pipeline with Transfer Learning & Augmentation
│   ├── evaluate.py             # Performance metrics, confusion matrix & classification report
│   └── disease_model.h5        # Trained Keras / TensorFlow model weights
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css       # Modern, responsive UI stylesheet
│   │   ├── js/
│   │   │   └── main.js         # Interactive upload, image preview & async requests
│   │   └── uploads/            # Temporary storage for user-uploaded scan images
│   ├── templates/
│   │   ├── index.html          # Dashboard & Image Upload Portal
│   │   ├── result.html         # Diagnosis, Confidence, Grad-CAM Heatmap & Treatment Plan
│   │   └── history.html        # Historical records of past diagnoses
│   ├── utils/
│   │   ├── preprocess.py       # Image resizing, tensor conversion, and normalization
│   │   └── grad_cam.py         # Gradient-weighted Class Activation Mapping for Explainable AI
│   ├── db.py                   # PostgreSQL connection & SQLAlchemy engine initialization
│   ├── models_db.py            # Database entities for scan history and disease reports
│   └── app.py                  # Flask application factory with routing & prediction logic
├── .env.example                # Template configuration for environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python**: 3.10+ recommended
- **PostgreSQL**: 14+ installed and running locally or on the cloud
- **Virtual Environment Tool**: `venv` or `conda`

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd arecanut-guard

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:
```bash
cp .env.example .env
```

### 4. Initialize Database
Ensure PostgreSQL is running and the database specified in `.env` exists:
```sql
CREATE DATABASE arecaguard_db;
```

When you start the Flask application, database tables (`scans`) are automatically created.

### 5. Training the Deep Learning Model
Organize your labeled images into `dataset/train/` and `dataset/val/` corresponding to the 5 disease classes. Then run:
```bash
python models/train_model.py
```
This will train the CNN model (e.g., MobileNetV2 / EfficientNet backbone) and save the weights to `models/disease_model.h5`.

To evaluate model metrics:
```bash
python models/evaluate.py
```

### 6. Launch the Web Application
```bash
python app/app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🔬 Core Features & Highlights

- **Deep Learning Inference**: Fast classification across 5 arecanut conditions.
- **Explainable AI (Grad-CAM)**: Visualizes the exact regions of the arecanut fruit that led to the model's prediction, building trust with farmers and agronomists.
- **Advisory & Treatment Engine**: Provides immediate chemical and organic control measures, recommended fungicide dosages (e.g., Bordeaux mixture for Mahali/Koleroga), and soil care tips.
- **Diagnostic History Tracking**: Stores all scans, confidence ratings, and timestamps in PostgreSQL for longitudinal farm health monitoring.
- **Modern Responsive UI**: Mobile-friendly interface optimized for field use.
