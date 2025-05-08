# 🎙️ Speech Emotion Recognition Using Deep Learning

This project focuses on detecting human emotions from speech using Mel-Frequency Cepstral Coefficients (MFCCs) and a Convolutional Neural Network (CNN). The system is trained and evaluated using the RAVDESS dataset and achieves high accuracy in classifying emotions such as **neutral**, **happy**, **sad**, and **angry**.
## 📌 Problem Statement
In human-computer interaction, understanding a speaker's emotion is crucial for improving responsiveness and empathy. However, many systems still lack the capability to interpret emotions from voice signals. This project addresses the challenge of emotion classification from speech, aiming to enhance AI-driven systems with emotional awareness.
## ✅ Proposed Solution
The system uses signal processing and deep learning techniques to classify emotions from raw speech data. The pipeline includes:
- Audio preprocessing (MFCC feature extraction)
- CNN-based emotion classification
- Evaluation using test accuracy and confusion matrix
## 🧠 Technologies & Libraries Used
- Python 3.8+
- PyTorch
- Librosa
- Scikit-learn
- NumPy
- Matplotlib
## 📂 Project Structure
Speech-emotion-recognition-Emo-DB-master/
├── data/
│ └── wav/
│ ├── train/
│ ├── valid/
│ └── test/
├── dataset.py
├── model.py
├── train.py
├── test.py
├── preprocess_ravdess.py
└── README.md
