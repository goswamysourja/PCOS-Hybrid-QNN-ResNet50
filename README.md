# 🩺 PCOS Detection using Hybrid Quantum Neural Networks and ResNet50

A Hybrid Quantum-Classical Deep Learning system for multi-class classification of ovarian ultrasound images using a pretrained ResNet50 backbone, classical feature compression, a four-qubit variational quantum circuit, and a classical neural classifier.

The project covers the complete machine learning lifecycle — from model development and training in Google Colab to Gradio-based inference, Docker containerization, Docker Hub image distribution, and cloud deployment on AWS EC2.

---

## 📌 Overview

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder that can involve abnormalities in ovarian morphology. Ultrasound imaging is one of the clinical tools used during evaluation, and automated image analysis can support research into computer-aided classification systems.

This project explores a **Hybrid Quantum-Classical Neural Network (QNN)** combined with **Deep Learning** for ovarian ultrasound image classification.

The system combines:

* Pretrained **ResNet50** for deep feature extraction
* Classical neural layers for feature compression
* A **four-qubit variational quantum circuit**
* **PennyLane** for quantum machine learning
* A classical neural classifier for final prediction
* **Gradio** for real-time inference
* **Docker** for application containerization
* **Docker Hub** for container image distribution
* **AWS EC2** for cloud deployment

The system classifies ovarian ultrasound images into five categories:

1. Complex Cyst
2. Dominant Follicle
3. Healthy
4. Poly Cyst
5. Simple Cyst

The deployed application allows users to upload an ovarian ultrasound image and receive a predicted class, confidence score, probability distribution, and application-level risk classification.

---

# ✨ Key Features

* Five-class ovarian ultrasound image classification
* Hybrid Quantum-Classical Neural Network
* Pretrained ResNet50 feature extraction
* Classical feature compression
* Four-qubit variational quantum circuit
* Angle Embedding
* Strongly Entangling Layers
* Pauli-Z expectation measurements
* Classical neural classification
* Confidence score generation
* Probability distribution visualization
* Risk classification
* Real-time image upload using Gradio
* Dockerized inference application
* Docker Hub image distribution
* AWS EC2 cloud deployment
* Ubuntu Linux server
* SSH-based remote deployment and management
* CPU-based inference deployment

---

# 🧠 System Architecture

```text
                    Ovarian Ultrasound Image
                              │
                              ▼
                     Image Preprocessing
                    Resize + Normalize
                              │
                              ▼
                    Pretrained ResNet50
                     Feature Extraction
                              │
                              ▼
                         2048 Features
                              │
                              ▼
                    Classical Compression
                         2048 → 64
                              │
                              ▼
                           64 → 4
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Quantum Layer       │
                  │                       │
                  │   4 Qubits            │
                  │   Angle Embedding     │
                  │   Strongly Entangling │
                  │   Layers              │
                  │   Pauli-Z Measurement │
                  └───────────┬───────────┘
                              │
                              ▼
                     4 Quantum Outputs
                              │
                              ▼
                    Classical Classifier
                         4 → 64 → 32 → 5
                              │
                              ▼
                     Five-Class Prediction
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
           Prediction     Confidence    Probability
                                            Chart
                              │
                              ▼
                     Risk Classification
```

---

# 📊 Dataset

## Ovarian Ultrasound Dataset

The primary dataset consists of **6,874 ovarian ultrasound images** distributed across five classes.

| Class             |    Images |
| ----------------- | --------: |
| Complex Cyst      |     1,386 |
| Dominant Follicle |     1,382 |
| Healthy           |     1,363 |
| Poly Cyst         |     1,368 |
| Simple Cyst       |     1,375 |
| **Total**         | **6,874** |

The dataset is used for multi-class ovarian ultrasound image classification.

---

## PCOS Clinical and Hormonal Dataset

A secondary clinical and hormonal dataset containing patient-related clinical, anthropometric, metabolic, and hormonal measurements is included in the research work.

This provides the foundation for potential multimodal analysis combining:

```text
Ultrasound Imaging
        +
Clinical / Hormonal Information
        ↓
Multimodal PCOS Analysis
```

The current Dockerized application focuses specifically on **ultrasound image classification**.

---

# 🔬 Model Architecture

## Stage 1 — Deep Feature Extraction

A pretrained **ResNet50** model from Torchvision is used as the primary feature extractor.

The model extracts high-level visual representations from ovarian ultrasound images.

```text
Input Ultrasound Image
          ↓
      ResNet50
          ↓
2048-dimensional feature vector
```

The extracted representation provides the input features for the subsequent classical and quantum processing stages.

---

## Stage 2 — Feature Compression

The 2048-dimensional ResNet50 representation is compressed before being passed to the quantum circuit.

```text
2048
 ↓
64
 ↓
4
```

The feature compression network consists of fully connected layers:

```text
ResNet50 Features
      │
      ▼
2048-dimensional vector
      │
      ▼
Linear Layer
2048 → 64
      │
      ▼
ReLU
      │
      ▼
Linear Layer
64 → 4
      │
      ▼
Quantum Layer
```

The final four-dimensional representation is used as the input to the four-qubit quantum circuit.

---

# ⚛️ Stage 3 — Quantum Neural Network

The quantum module consists of a **four-qubit variational quantum circuit** implemented using **PennyLane**.

### Feature Encoding

The four classical features are encoded into the quantum circuit using:

```text
4 Classical Features
        ↓
Angle Embedding
        ↓
RY Rotations
        ↓
4 Qubits
```

### Trainable Quantum Operations

The circuit uses:

```text
Strongly Entangling Layers
```

to introduce trainable quantum operations and interactions between the qubits.

### Measurement

The quantum circuit produces four outputs through expectation-value measurements:

```text
Pauli-Z Expectation Values
          ↓
     4 Quantum Outputs
```

These outputs are then passed to the classical classification network.

---

# 🧮 Stage 4 — Classical Classification

The four quantum outputs are processed through a fully connected neural classifier:

```text
4
↓
64
↓
32
↓
5
```

The final five output classes are:

| Output | Class             |
| -----: | ----------------- |
|      1 | Complex Cyst      |
|      2 | Dominant Follicle |
|      3 | Healthy           |
|      4 | Poly Cyst         |
|      5 | Simple Cyst       |

The classifier produces class probabilities, from which the final prediction and confidence score are obtained.

---

# 🩻 Risk Classification

The deployed application provides an additional application-level risk classification based on the predicted class.

### Higher-Risk Category

* Complex Cyst
* Poly Cyst
* Simple Cyst

### Lower-Risk Category

* Dominant Follicle
* Healthy

This classification is implemented as part of the application's output logic and is separate from the underlying five-class model prediction.

---

# 🔍 Explainable AI

The original research implementation explored **Explainable AI (XAI)** using **LIME — Local Interpretable Model-Agnostic Explanations**.

LIME was used to investigate which regions of an ultrasound image contributed most to an individual model prediction.

This research component provided:

* Visual interpretation of predictions
* Identification of influential image regions
* Improved understanding of model behavior
* Additional model validation capability
* Greater prediction transparency

### Current Docker Deployment

The current lightweight Dockerized inference application does **not** include LIME.

The deployed application focuses on:

```text
Image
 ↓
Prediction
 ↓
Confidence Score
 ↓
Probability Distribution
 ↓
Risk Classification
```

LIME remains part of the original research work, while the deployed version was streamlined for lightweight cloud inference on the available EC2 resources.

---

# 🖥️ Real-Time Prediction Interface

The inference application uses **Gradio** to provide a web-based interface.

## Workflow

```text
Upload Ultrasound Image
          ↓
Image Preprocessing
          ↓
ResNet50 Feature Extraction
          ↓
Feature Compression
          ↓
Quantum Neural Network
          ↓
Classical Classifier
          ↓
Prediction
          ↓
Confidence Score
          ↓
Risk Classification
          ↓
Probability Distribution
```

The interface provides:

* Predicted class
* Confidence percentage
* Risk classification
* Probability distribution chart

Example:

```text
Prediction: Healthy

Confidence: XX.XX%

Risk Classification: LOW RISK
```

---

# 🐳 Docker Deployment

The Gradio inference application was containerized using **Docker**.

Docker packages the application, Python environment, machine learning libraries, quantum computing dependencies, model, and inference code into a reproducible container.

## Docker Architecture

```text
PCOS Application
       │
       ▼
   Dockerfile
       │
       ▼
   Docker Image
       │
       ▼
 Docker Container
       │
       ▼
 Gradio Application
       │
       ▼
    Port 7860
```

### Build the Docker Image

```bash
docker build -t pcos-app .
```

### Run the Container

```bash
docker run -d -p 7860:7860 --name pcos-container pcos-app
```

The application can be tested locally using:

```text
http://localhost:7860
```

---

# 📦 Docker Hub

The Docker image was pushed to Docker Hub for image distribution and cloud deployment.

### Docker Hub Repository

**Repository:** `soumik467/pcos-app`

**Docker Hub:**
[Add your Docker Hub repository link here]

The image can be pulled using:

```bash
docker pull soumik467/pcos-app:latest
```

This allows the same containerized application to be deployed on another machine or cloud server without rebuilding the image from the source code.

The deployment workflow is:

```text
Local Development
       ↓
Docker Build
       ↓
Docker Image
       ↓
Docker Hub
       ↓
docker pull
       ↓
AWS EC2
```

---

# ☁️ AWS EC2 Deployment

The Dockerized PCOS application was deployed to an **AWS EC2** virtual machine.

## Deployment Architecture

```text
                    Docker Hub
                        │
                        │ docker pull
                        ▼
                 AWS EC2 Instance
                 Ubuntu 24.04 LTS
                        │
                        ▼
                  Docker Engine
                        │
                        ▼
                 Docker Container
                  pcos-container
                        │
                        ▼
                 Gradio Application
                        │
                     Port 7860
                        │
                        ▼
                  Public IPv4
                        │
                        ▼
                    Web Browser
```

---

# 🖥️ EC2 Configuration

The application was deployed using:

| Configuration    | Value                   |
| ---------------- | ----------------------- |
| Cloud Provider   | AWS                     |
| Service          | Amazon EC2              |
| Region           | Asia Pacific (Mumbai)   |
| Operating System | Ubuntu Server 24.04 LTS |
| Instance Type    | t3.micro                |
| Storage          | 30 GB gp3               |
| Application Port | 7860                    |
| SSH Port         | 22                      |

---

# 🔐 SSH-Based Deployment

SSH was used to remotely connect to the Ubuntu EC2 server.

Example:

```bash
ssh -i pcos-key.pem ubuntu@<EC2_PUBLIC_IP>
```

After connecting to the server, Docker was installed and configured.

The Docker image was then pulled from Docker Hub:

```bash
docker pull soumik467/pcos-app:latest
```

The application container was started using:

```bash
docker run -d \
  -p 7860:7860 \
  --name pcos-container \
  soumik467/pcos-app:latest
```

The running container can be checked using:

```bash
docker ps
```

---

# 🌐 Cloud Application Access

The Gradio application is exposed through port **7860**.

The deployed application can be accessed through:

```text
http://<EC2_PUBLIC_IP>:7860
```

### Live Application

[Add your current EC2 application link here]

The EC2 Security Group is configured to allow inbound traffic to port **7860**, enabling access to the Gradio interface through the instance's public IPv4 address.

---

# 🚀 Complete Project Lifecycle

The complete development and deployment workflow is:

```text
                         DATASET
                            │
                            ▼
                    Google Colab
                            │
                            ▼
                     Model Training
                            │
                            ▼
                  ResNet50 + Hybrid QNN
                            │
                            ▼
                      Trained Model
                            │
                            ▼
                  Gradio Inference App
                            │
                            ▼
                       Dockerfile
                            │
                            ▼
                      Docker Build
                            │
                            ▼
                     Docker Image
                            │
                            ▼
                      Docker Hub
                            │
                       docker pull
                            ▼
                       AWS EC2
                            │
                            ▼
                     Ubuntu Server
                            │
                            ▼
                      Docker Engine
                            │
                            ▼
                     Docker Container
                            │
                            ▼
                         Gradio
                            │
                            ▼
                       Port 7860
                            │
                            ▼
                  Public Web Application
```

This represents the complete transition from **research and model development → application development → containerization → cloud deployment**.

---

# 🧰 Technologies Used

## Machine Learning & Deep Learning

* Python
* PyTorch
* Torchvision
* ResNet50
* Neural Networks
* Computer Vision
* Image Classification

## Quantum Machine Learning

* PennyLane
* Quantum Neural Networks
* Variational Quantum Circuits
* Angle Embedding
* Strongly Entangling Layers
* Pauli-Z Measurements

## Explainable AI

* LIME
* Explainable AI research

## Application

* Gradio
* NumPy
* Matplotlib
* Pillow

## Deployment & Cloud

* Docker
* Docker Hub
* AWS EC2
* Ubuntu Linux
* SSH

## Development

* Google Colab
* Visual Studio Code

---

# 📈 Results

The project demonstrates a complete hybrid quantum-classical image classification and deployment pipeline.

The system demonstrates:

* Five-class ovarian ultrasound classification
* ResNet50-based deep feature extraction
* Classical feature compression
* Four-qubit quantum processing
* Hybrid quantum-classical classification
* Confidence-based prediction
* Probability distribution visualization
* Real-time inference through Gradio
* Dockerized application deployment
* Docker Hub image distribution
* AWS EC2 cloud deployment

The research implementation additionally explored **LIME-based explainability** for interpreting individual image predictions.

---

# 👨‍💻 Team

## Developers

* **Sourja Goswamy**
* **Soumik Chowdhury**

## Research Domains

* Healthcare AI
* Medical Imaging
* Deep Learning
* Computer Vision
* Quantum Machine Learning
* Explainable AI
* Cloud Deployment
