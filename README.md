# 🔍 AI-Based Manufacturing Defect Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌐 Live Demo

🚀 **Try the Application Here:**

**https://ai-based-manufacturing-defect-detection-ai.streamlit.app/**

---

# 📌 Project Overview

AI-Based Manufacturing Defect Detection is a Deep Learning application that automatically identifies surface defects in steel sheets using a Convolutional Neural Network (CNN).

The model is trained on the **NEU Surface Defect Database (NEU-DET)** and classifies uploaded steel surface images into one of six manufacturing defect categories.

The application is deployed using **Streamlit Cloud**, enabling real-time defect detection through a clean and interactive web interface.

---

# 🎯 Features

- 🔍 Detects manufacturing defects from steel surface images
- 🧠 CNN-based Deep Learning model
- 📷 Image upload interface
- ⚡ Real-time prediction
- 📊 Prediction confidence score
- ⏱ Fast inference time
- 📈 Probability distribution visualization
- ☁️ Deployed on Streamlit Cloud
- 📥 Automatic model download using Google Drive

---

# 🏭 Supported Defect Classes

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

---

# 🧠 Deep Learning Pipeline

```
Steel Surface Image
        │
        ▼
Image Upload
        │
        ▼
Image Preprocessing
(224 × 224 RGB)
        │
        ▼
CNN Model
(TensorFlow / Keras)
        │
        ▼
Softmax Classification
        │
        ▼
Prediction + Confidence Score
```

---

# 📂 Dataset

**Dataset:** NEU Surface Defect Database (NEU-DET)

- 6 defect categories
- Grayscale steel surface images
- Widely used benchmark dataset for industrial defect detection

---

# 🛠 Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Pillow
- Matplotlib
- Scikit-learn
- gdown

---

# 📊 Model Information

| Parameter | Value |
|-----------|-------|
| Model | Convolutional Neural Network (CNN) |
| Framework | TensorFlow / Keras |
| Dataset | NEU-DET |
| Image Size | 224 × 224 |
| Validation Accuracy | **83.3%** |

---

# 📸 Application Preview

### Home Page

Upload a steel surface image and receive an instant prediction with confidence score and probability distribution.

---

# 📁 Project Structure

```
AI-Based-Manufacturing-Defect-Detection/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── manufacturing_defect_cnn.keras
├── AI-Based Manufacturing Defect Detection using Deep Learning.ipynb
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Ojas4git/AI-Based-Manufacturing-Defect-Detection.git
```

Move into the project directory

```bash
cd AI-Based-Manufacturing-Defect-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📈 Sample Output

- ✅ Predicted Defect
- ✅ Confidence Score
- ✅ Inference Time
- ✅ Probability Distribution
- ✅ Defect Description

---

# 💡 Future Improvements

- Transfer Learning using EfficientNet or ResNet
- Explainable AI (Grad-CAM)
- Batch Image Prediction
- PDF Prediction Reports
- REST API Deployment
- Mobile-Friendly Interface

---

# 👨‍💻 Developer

**Ojas Savkar**

Mechanical Engineering Undergraduate | Artificial Intelligence | Deep Learning

GitHub:
https://github.com/Ojas4git

---

# ⭐ If you found this project useful

Please consider giving the repository a **Star ⭐**
