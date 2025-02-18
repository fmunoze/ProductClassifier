from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# Definición de la clase ProductClassifier
class ProductClassifier(nn.Module):
    def __init__(self):
        super(ProductClassifier, self).__init__()

        # Dos bloques convolucionales simples
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Un solo batch normalization
        self.bn = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Una capa fully connected más pequeña
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 clases

        # Un solo dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Primer bloque convolucional
        x = self.pool(F.relu(self.conv1(x)))

        # Segundo bloque convolucional con batch norm
        x = self.pool(F.relu(self.bn(self.conv2(x))))

        # Aplanar
        x = x.view(-1, 64 * 56 * 56)

        # Fully connected con dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Cargar el modelo entrenado
model = ProductClassifier()  # Ahora la clase está definida
model.load_state_dict(torch.load("ecommerce_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Definir las transformaciones necesarias para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Clases del modelo
classes = ['jeans', 'sofa', 'tshirt', 'tv']

@app.route('/')
def home():
    # Renderizar la plantilla index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Convertir la imagen a un formato que pueda ser procesado por el modelo
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = transform(image).unsqueeze(0)  # Añadir una dimensión de batch

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

        # Obtener la clase predicha y la confianza
        predicted_class = classes[predicted.item()]
        confidence_value = confidence[predicted.item()].item()

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence_value:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)