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

# Estilo personalizado con HTML y CSS en Streamlit
st.markdown("""
    <style>
        /* Estilo general */
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background-color: #1E1E1E;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
        .upload-area {
            border: 2px dashed #4CAF50;
            padding: 20px;
            margin: 20px 0;
            cursor: pointer;
            border-radius: 8px;
            background-color: #2E2E2E;
            color: #FAFAFA;
        }
        .upload-area:hover {
            background-color: #3E3E3E;
        }
        #preview {
            max-width: 100%;
            height: auto;
            margin-top: 15px;
            border-radius: 5px;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
        }
        #result {
            margin-top: 20px;
            font-size: 1.2em;
            padding: 10px;
            background-color: #2E2E2E;
            border-radius: 5px;
            color: #FAFAFA;
        }
        .btn-primary {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            cursor: pointer;
            font-size: 1em;
        }
        .btn-primary:hover {
            background-color: #45a049;
        }
        .btn-secondary {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            cursor: pointer;
            font-size: 1em;
        }
        .btn-secondary:hover {
            background-color: #5a6268;
        }
        .hidden {
            display: none;
        }
        h1 {
            color: #4CAF50;
        }
        p {
            color: #FAFAFA;
        }
    </style>
""", unsafe_allow_html=True)

# Interfaz de Streamlit
st.markdown("""
    <div class="container">
        <h1>Clasificador de Productos</h1>
        <p>Sube una imagen de un producto para clasificarlo</p>
    </div>
""", unsafe_allow_html=True)

# Subir una imagen
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="file-uploader")

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Realizar la predicción
    if st.button("🔍 Predecir", key="predict-btn"):
        try:
            image = transform(image).unsqueeze(0)  # Añadir una dimensión de batch

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

            # Obtener la clase predicha y la confianza
            predicted_class = classes[predicted.item()]
            confidence_value = confidence[predicted.item()].item()

            st.markdown(f"""
                <div id="result">
                    <p><strong>Categoría:</strong> <span id="predicted-class">{predicted_class}</span></p>
                    <p><strong>Confianza:</strong> <span id="confidence">{confidence_value:.2f}%</span></p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Botón para resetear
    if st.button("🔄 Subir otra imagen", key="reset-btn"):
        st.experimental_rerun()