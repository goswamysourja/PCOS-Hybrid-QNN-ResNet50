# 🩺 PCOS Detection using Hybrid Quantum Neural Networks and Deep Learning

## Overview

Polycystic Ovary Syndrome (PCOS) is one of the most common hormonal disorders affecting women worldwide. Early diagnosis often requires the analysis of ovarian ultrasound scans alongside clinical and hormonal indicators, making the diagnostic process time-consuming and highly dependent on specialist expertise.

This project presents a Hybrid Quantum Neural Network (QNN) integrated with Deep Learning and Explainable AI to classify ovarian ultrasound images into multiple clinically relevant categories. The system combines a pretrained ResNet50 feature extractor, a variational quantum circuit, a classical neural classifier, and LIME-based explainability to deliver accurate and interpretable predictions. A Gradio-powered real-time interface allows users to upload ultrasound images and instantly receive classification results along with confidence scores and probability distributions.

---

## Key Features

- Five-class ovarian ultrasound classification
- Hybrid Quantum-Classical Neural Network architecture
- Explainable AI using LIME
- Real-time image upload and prediction
- Confidence score generation
- Probability distribution visualization
- Research-oriented architecture integrating Quantum Machine Learning

---

## Dataset

### Ovarian Ultrasound Dataset (Ovarian_US)

The primary dataset consists of 6,874 ovarian ultrasound images distributed across five classes:

| Class | Images |
|---------|---------:|
| Complex Cyst | 1,386 |
| Dominant Follicle | 1,382 |
| Healthy | 1,363 |
| Poly Cyst | 1,368 |
| Simple Cyst | 1,375 |
| **Total** | **6,874** |

### PCOS Clinical and Hormonal Dataset

The secondary dataset contains patient clinical, anthropometric, metabolic, and hormonal measurements for PCOS and Non-PCOS classification. This dataset is included for multimodal PCOS research and future integration with imaging features.

---

## System Pipeline

```text
Ultrasound Image
        │
        ▼
Data Preprocessing
(Resize + Normalize)
        │
        ▼
Pretrained ResNet50
Feature Extraction
        │
        ▼
Feature Compression
2048 → 64 → 4
        │
        ▼
Variational Quantum Circuit
(4 Qubits)
        │
        ▼
Quantum Measurements
        │
        ▼
Classical Neural Network
        │
        ▼
5-Class Prediction
        │
        ▼
LIME Explainability
```

---

## Model Architecture

### Stage 1: Deep Feature Extraction

A pretrained ResNet50 backbone is used to extract hierarchical spatial and morphological features from ovarian ultrasound scans. The network learns representative patterns related to ovarian structures, follicles, cyst formations, and tissue abnormalities.

Output:

```text
2048-dimensional feature vector
```

### Stage 2: Feature Compression

To interface classical deep learning with quantum computation, the extracted feature vector is projected into a lower-dimensional latent representation:

```text
2048 → 64 → 4
```

This compressed representation becomes the input to the quantum circuit.

### Stage 3: Quantum Layer

The quantum module consists of a four-qubit variational quantum circuit implemented using PennyLane.

Feature Encoding:

```text
Angle Embedding (RY Rotations)
```

Trainable Quantum Operations:

```text
Strongly Entangling Layers
```

Measurement:

```text
Pauli-Z Expectation Values
```

The quantum circuit learns complex non-linear feature interactions that may not be efficiently captured by classical neural networks alone.

### Stage 4: Classical Classification

The quantum outputs are processed through a fully connected classifier:

```text
4 → 64 → 32 → 5
```

Final output classes:

1. Complex Cyst
2. Dominant Follicle
3. Healthy
4. Poly Cyst
5. Simple Cyst

---

## Explainable AI (XAI)

Medical AI systems require transparency and interpretability. To improve trust and explain predictions, this project incorporates Local Interpretable Model-Agnostic Explanations (LIME).

LIME highlights the regions of an ultrasound image that contribute most significantly to the model’s prediction, allowing visual verification of model decisions and reducing black-box behavior.

Benefits:

- Visual explanation of predictions
- Improved clinician confidence
- Better model validation
- Enhanced interpretability

---

## Real-Time Prediction Interface

A Gradio-based web interface enables real-time inference.

Workflow:

1. Upload ovarian ultrasound image
2. Automatic preprocessing
3. Hybrid QNN inference
4. Confidence score calculation
5. Probability distribution generation
6. Final class prediction

The interface provides an intuitive and interactive way to evaluate ultrasound images without requiring technical expertise.

---

## Technologies Used

- PyTorch
- Torchvision
- PennyLane
- Quantum Neural Networks (QNN)
- ResNet50
- LIME
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Pillow (PIL)
- Gradio
- Google Colab

---

## Results

The proposed Hybrid Quantum Neural Network successfully classifies ovarian ultrasound images into five clinically relevant categories while maintaining strong predictive performance and model interpretability.

Key outcomes include:

- High-confidence classification across five classes
- Successful integration of quantum and classical learning
- Explainable predictions through LIME visualization
- Real-time deployment using Gradio

---

## Future Enhancements

- Multimodal fusion of ultrasound and clinical datasets
- Advanced quantum circuit optimization
- Cloud-based deployment
- Mobile application integration
- Clinical decision support dashboard
- Automated medical report generation

---

## Team

### Developers

- Sourja Goswamy
- Soumik Chowdhury

### Research Domain

Healthcare AI • Medical Imaging • Quantum Machine Learning • Explainable AI

---

## License

This project is intended for academic research and educational purposes. Clinical usage requires extensive validation and regulatory approval.