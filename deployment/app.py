
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

from PIL import Image
from torchvision import models, transforms
import pennylane as qml
from pennylane.qnn import TorchLayer


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "best_model.pth"

IMG_SIZE = 128
N_QUBITS = 4

device = torch.device("cpu")


# ============================================================
# 2. CLASS NAMES
# ============================================================
# These are the exact classes from your ImageFolder dataset.
# ImageFolder sorts class names alphabetically.

classes = [
    "complex_cyst",
    "dominant_follicle",
    "healthy",
    "poly_cyst",
    "simple_cyst"
]


# ============================================================
# 3. QUANTUM DEVICE
# ============================================================

try:
    dev = qml.device(
        "lightning.qubit",
        wires=N_QUBITS
    )
except Exception:
    dev = qml.device(
        "default.qubit",
        wires=N_QUBITS
    )


# ============================================================
# 4. QUANTUM CIRCUIT
# ============================================================

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    qml.templates.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS)
    )

    qml.templates.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    ]


weight_shapes = {
    "weights": (2, N_QUBITS, 3)
}

qnn_layer = TorchLayer(
    quantum_circuit,
    weight_shapes
)


# ============================================================
# 5. MODEL ARCHITECTURE
# ============================================================

class HybridQNNModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # Same ResNet50 used during training
        self.resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Freeze ResNet backbone
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace original ResNet classifier
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_QUBITS)
        )

        # Quantum layer
        self.qnn = qnn_layer

        # Same classifier used during training
        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        x = self.resnet(x)
        x = self.qnn(x)
        x = self.classifier(x)

        return x


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = HybridQNNModel(
    num_classes=len(classes)
)

# Load your trained model
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

# Evaluation mode
model.eval()


# ============================================================
# 7. IMAGE TRANSFORMATION
# ============================================================
# This is the same validation/inference transformation
# used in your original code.

val_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 8. RISK CLASSIFICATION
# ============================================================

pcos_classes = {
    "poly_cyst",
    "complex_cyst",
    "simple_cyst"
}


# ============================================================
# 9. PREDICTION FUNCTION
# ============================================================

def analyze_image(image):

    if image is None:
        return "Please upload an ultrasound image.", None

    try:

        # Convert image to RGB
        image = image.convert("RGB")

        # Apply preprocessing
        input_tensor = val_transform(image)

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(device)

        # Disable gradient calculation
        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        # Get predicted class
        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_class = classes[predicted_index]

        confidence = probabilities[
            predicted_index
        ].item() * 100

        # Risk classification
        if predicted_class in pcos_classes:
            risk = "HIGH RISK"
        else:
            risk = "LOW RISK"

        # ====================================================
        # Probability chart
        # ====================================================

        probs = probabilities.cpu().numpy() * 100

        plt.figure(figsize=(8, 4))

        plt.bar(
            classes,
            probs
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.ylabel("Probability (%)")
        plt.title("Prediction Probabilities")

        plt.tight_layout()

        chart_path = "/tmp/probability_chart.png"

        plt.savefig(chart_path)
        plt.close()

        # ====================================================
        # Result text
        # ====================================================

        result = f"""
Prediction: {predicted_class}

Confidence: {confidence:.2f}%

Risk Classification: {risk}

Note: This system is for research/educational purposes
and should not be considered a medical diagnosis.
"""

        return result, chart_path

    except Exception as e:

        return f"Error: {str(e)}", None


# ============================================================
# 10. GRADIO INTERFACE
# ============================================================

demo = gr.Interface(

    fn=analyze_image,

    inputs=gr.Image(
        type="pil",
        label="Upload Ultrasound Image"
    ),

    outputs=[
        gr.Textbox(
            label="Analysis Result"
        ),

        gr.Image(
            label="Probability Chart"
        )
    ],

    title="PCOS Detection using Hybrid QNN + ResNet50",

    description=(
        "Upload an ovarian ultrasound image "
        "to classify it using the trained "
        "Hybrid QNN + ResNet50 model."
    )
)


# ============================================================
# 11. START GRADIO
# ============================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)

