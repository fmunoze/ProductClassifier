import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F

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

# Interfaz de Streamlit
st.title("Clasificador de Productos")

# Subir una imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Realizar la predicción
    try:
        image = transform(image).unsqueeze(0)  # Añadir una dimensión de batch

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

        # Obtener la clase predicha y la confianza
        predicted_class = classes[predicted.item()]
        confidence_value = confidence[predicted.item()].item()

        st.success(f"Predicción: {predicted_class} (Confianza: {confidence_value:.2f}%)")

    except Exception as e:
        st.error(f"Error: {str(e)}")