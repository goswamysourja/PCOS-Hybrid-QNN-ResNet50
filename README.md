# 🩺 PCOS Detection using Hybrid Quantum Neural Networks and ResNet50

A Hybrid Quantum-Classical Deep Learning system for multi-class classification of ovarian ultrasound images using a pretrained ResNet50 backbone, classical feature compression, a four-qubit variational quantum circuit, and a classical neural classifier.

The project covers the complete machine learning lifecycle — from model development and training in Google Colab to Gradio-based inference, Docker containerization, Docker Hub image distribution, and cloud deployment on AWS EC2.

---

# 📌 Overview

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder that can involve abnormalities in ovarian morphology. Ultrasound imaging is one of the clinical tools used during evaluation, and automated image analysis can support research into computer-aided classification systems.

This project explores a **Hybrid Quantum-Classical Neural Network (QNN)** combined with **Deep Learning** for ovarian ultrasound image classification.

The system combines:

- Pretrained **ResNet50** for deep feature extraction
- Classical neural layers for feature compression
- A **four-qubit variational quantum circuit**
- **PennyLane** for quantum machine learning
- A classical neural classifier for final prediction
- **Gradio** for real-time inference
- **Docker** for application containerization
- **Docker Hub** for container image distribution
- **AWS EC2** for cloud deployment

The system classifies ovarian ultrasound images into five categories:

1. Complex Cyst
2. Dominant Follicle
3. Healthy
4. Poly Cyst
5. Simple Cyst

The deployed application allows users to upload an ovarian ultrasound image and receive a predicted class, confidence score, probability distribution, and application-level risk classification.

---

# ✨ Key Features

- Five-class ovarian ultrasound image classification
- Hybrid Quantum-Classical Neural Network
- Pretrained ResNet50 feature extraction
- Classical feature compression
- Four-qubit variational quantum circuit
- Angle Embedding
- Strongly Entangling Layers
- Pauli-Z expectation measurements
- Classical neural classification
- Confidence score generation
- Probability distribution visualization
- Risk classification
- Real-time image upload using Gradio
- Dockerized inference application
- Docker Hub image distribution
- AWS EC2 cloud deployment
- Ubuntu Linux server
- SSH-based remote deployment and management
- CPU-based inference deployment

---

# 🧠 System Architecture

```mermaid
flowchart TD
    A[Ovarian Ultrasound Image]
    --> B[Image Preprocessing<br/>Resize + Normalize]

    B --> C[Pretrained ResNet50<br/>Feature Extraction]

    C --> D[2048 Features]

    D --> E[Linear Layer<br/>2048 → 64]

    E --> F[ReLU]

    F --> G[Linear Layer<br/>64 → 4]

    G --> H[Quantum Layer<br/>4 Qubits]

    H --> I[4 Quantum Outputs]

    I --> J[Classical Classifier<br/>4 → 64 → 32 → 5]

    J --> K[Five-Class Prediction]

    K --> L[Prediction]
    K --> M[Confidence]
    K --> N[Probability Distribution]
    K --> O[Risk Classification]
