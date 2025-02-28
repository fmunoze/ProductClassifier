import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

# Definición de la clase ResNet para clasificación de productos
class ResNetProductClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetProductClassifier, self).__init__()
        # Cargar ResNet50 preentrenado
        self.resnet = models.resnet50(pretrained=False)
        
        # Modificar la capa final para nuestro problema de clasificación
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

# Clases del modelo
classes = ['jeans', 'sofa', 'tshirt', 'tv']

# Función para cargar y preparar el modelo
@st.cache_resource
def load_model():
    model = ResNetProductClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load("resnet_ecommerce_classifier.pth", map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

# Definir las transformaciones necesarias para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función para aplicar predicción
def predict(image, model):
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
    return {
        'class': classes[predicted.item()],
        'confidence': probabilities[predicted.item()].item(),
        'all_probabilities': {classes[i]: probabilities[i].item() for i in range(len(classes))}
    }

# Función para mostrar la predicción animada
def animated_prediction(result):
    progress_text = "Analizando imagen..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(101):
        time.sleep(0.01)
        my_bar.progress(percent_complete, text=progress_text)
    
    st.success("¡Análisis completado!")
    
    # Mostrar la categoría predicha con estilo
    st.markdown(f"""
    <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1E90FF;'>
        <h2 style='margin-top: 0; color: #1E90FF;'>Resultado del Análisis</h2>
        <h3>Categoría: <span style='color: #1E90FF; font-weight: bold;'>{result['class'].upper()}</span></h3>
        <h4>Confianza: <span style='color: #1E90FF;'>{result['confidence']:.2f}%</span></h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar gráfico de confianza para todas las clases
    confidences = result['all_probabilities']
    
    # Ordenar los valores de confianza de mayor a menor
    sorted_confidences = {k: v for k, v in sorted(confidences.items(), key=lambda item: item[1], reverse=True)}
    
    # Crear barras coloridas para cada clase
    for cls, conf in sorted_confidences.items():
        # Color más intenso para la clase predicha
        color = "#1E90FF" if cls == result['class'] else "#A9A9A9"
        st.markdown(f"""
        <div style='margin-bottom: 10px;'>
            <div style='display: flex; align-items: center;'>
                <div style='width: 100px; text-align: right; margin-right: 10px;'>{cls}</div>
                <div style='flex-grow: 1; background-color: #f0f0f0; border-radius: 5px; height: 25px;'>
                    <div style='width: {conf}%; height: 100%; background-color: {color}; border-radius: 5px; display: flex; align-items: center; justify-content: flex-end;'>
                        <span style='margin-right: 5px; color: white; font-weight: bold;'>{conf:.1f}%</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Estilo personalizado con HTML y CSS en Streamlit
st.markdown("""
    <style>
        /* Estilo general */
        .main {
            background-color: #f8f9fa;
        }
        h1 {
            color: #1E90FF;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-align: center;
        }
        h2 {
            color: #333;
            font-size: 1.8rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .subtitle {
            color: #555;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-container {
            background-color: #fff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0078FF;
        }
        .stImage {
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Animación de carga */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .loading {
            animation: pulse 1.5s infinite;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 0.8rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            h1 {
                font-size: 2rem;
            }
            .subtitle {
                font-size: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Interfaz de Streamlit
st.markdown("<h1>Clasificador Avanzado de Productos</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by ResNet: Sube una imagen para clasificar el producto</p>", unsafe_allow_html=True)

with st.expander("ℹ️ Acerca de esta app", expanded=False):
    st.markdown("""
    Esta aplicación utiliza un modelo de deep learning ResNet50 para clasificar imágenes de productos en cuatro categorías:
    - 👖 Jeans
    - 🛋️ Sofás
    - 👕 Camisetas
    - 📺 Televisores
    
    El modelo ha sido entrenado con técnicas avanzadas de transfer learning para ofrecer mayor precisión.
    
    **¿Cómo usar?**
    1. Sube una imagen de un producto
    2. Haz clic en "Analizar Imagen"
    3. Observa los resultados
    """)

# Cargar el modelo
try:
    with st.spinner("Cargando modelo..."):
        model = load_model()
    st.success("¡Modelo cargado correctamente!")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Área principal
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    
    # Subir una imagen
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imagen subida', use_column_width=True)
        
        # Botón para realizar la predicción
        if st.button("🔍 Analizar Imagen"):
            try:
                # Realizar la predicción
                result = predict(image, model)
                
                # Mostrar resultado animado
                animated_prediction(result)
                
            except Exception as e:
                st.error(f"Error durante la predicción: {str(e)}")
    else:
        # Mostrar mensaje cuando no hay imagen
        st.info("👆 Sube una imagen para comenzar")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.markdown("### Ejemplos de productos")
    
    # Ejemplos de categorías
    st.markdown("""
    * **Jeans**: Pantalones vaqueros, jeans, denim
    * **Sofa**: Sofás, sillones, muebles de sala
    * **T-shirt**: Camisetas, polos, tops
    * **TV**: Televisores, monitores, pantallas
    """)
    
    st.markdown("### Consejos para mejores resultados")
    st.markdown("""
    * Usa imágenes de buena calidad
    * Asegúrate que el producto esté claramente visible
    * Evita imágenes con múltiples productos
    * El fondo simple mejora la precisión
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 E-commerce Product Classifier | Powered by ResNet</div>", unsafe_allow_html=True)