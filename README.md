# 🍎 Apple Leaf Disease Detection – Multi-Language AI Web Application

A **Deep Learning–based web application** that detects **Apple leaf diseases from images** and provides **precautions, fertilizers, and pesticide recommendations** in **English, Hindi, and Telugu** for better farmer and user understanding.

This project uses a **CNN model (EfficientNet-B3)** trained on an Apple Disease dataset and is deployed using **Flask + Docker**.

---

## 🚀 Features

- 🍃 Detects **4 Apple Leaf Conditions**
  - Apple Scab
  - Black Rot
  - Cedar Apple Rust
  - Healthy Leaf
- 🌍 Multi-language support:
  - 🇬🇧 English  
  - 🇮🇳 Hindi  
  - 🇮🇳 Telugu  
- 📊 Displays **confidence scores for all classes**
- 🧪 Provides:
  - Precautions
  - Fertilizers
  - Pesticides
- 🐳 Fully **Dockerized**
- ☁️ Production-ready with **Gunicorn**
- 🧠 EfficientNet-B3 based Deep Learning model

---

## 🧠 Model Information

- **Architecture**: EfficientNet-B3  
- **Input Size**: `300 × 300 × 3`  
- **Framework**: TensorFlow / Keras  
- **Classes Mapping**:
  ```text
  {
    'apple_scab': 0,
    'black_rot': 1,
    'cedar_apple_rust': 2,
    'healthy': 3
  }




PROJECT STRUCTURE

apple_disease_detection/
│
├── app.py                  # Flask backend
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── apple_disease_model.h5  # Trained model (Git LFS)
├── templates/              # HTML files
├── static/                 # CSS, JS, images
├── uploads/                # Uploaded images (runtime)
├── README.md               # Documentation
└── .gitattributes          # Git LFS config


Model File & Git LFS

git lfs install
git clone <your-github-repo-url>
cd apple_disease_detection



# 🖥️ Run Locally (Without Docker)

## 1️⃣ Create Virtual Environment
python -m venv venv
### Windows
venv\Scripts\activate

### Linux / Mac
source venv/bin/activate

## 2️⃣ Install Dependencies
pip install -r requirements.txt

## 3️⃣ Run the Application
python app.py

## 4️⃣ Open Browser
http://localhost:5000


# 🐳 Run Using Docker (Recommended)

## 1️⃣ Build Docker Image
docker build -t apple-crop-disease .

## 2️⃣ Run Docker Container
docker run -p 5000:10000 apple-crop-disease

## 3️⃣ Open Browser   
http://localhost:5000



# ABOUT BROWSER
<!-- 
🌍 Language Support

The application supports:

    English

    Hindi

    Telugu

All outputs (disease name, precautions, fertilizers, pesticides, probabilities) are translated and localized. -->


# 📷 How to Use the Application

Open the web application

Upload an Apple leaf image

Select preferred language

Click Predict

    View:

        1.Detected disease

        2.Confidence percentage

        3.Precautions

        4.Fertilizers

        5.Pesticides

        6.All class probabilities