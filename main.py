import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import time

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Productos",
    page_icon="🛍️",
    layout="wide"
)

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

# Estilo personalizado con CSS
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
        h3 {
            color: #1E90FF;
            font-size: 1.3rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .subtitle {
            color: #555;
            font-size: 1.2rem;
            text-align: center;
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
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 0.8rem;
        }
        
        /* Columna vacía */
        .empty-column {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            background-color: #f8f9fa;
            border-radius: 10px;
            border: 2px dashed #ccc;
            color: #666;
            text-align: center;
            padding: 20px;
        }
        
        /* Contenedor de resultados */
        .result-box {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #1E90FF;
        }
        
        /* Barras de confianza */
        .confidence-bar-container {
            margin-bottom: 10px;
        }
        .confidence-bar-label {
            width: 100px;
            text-align: right;
            margin-right: 10px;
            display: inline-block;
        }
        .confidence-bar-outer {
            flex-grow: 1;
            display: inline-block;
            width: calc(100% - 120px);
            background-color: #f0f0f0;
            border-radius: 5px;
            height: 25px;
            vertical-align: middle;
        }
        .confidence-bar-inner {
            height: 100%;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
        }
        .confidence-bar-text {
            margin-right: 5px;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown("<h1>Clasificador Avanzado de Productos</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by ResNet: Sube una imagen para clasificar el producto</p>", unsafe_allow_html=True)

# Información de la aplicación
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

# Inicializar el estado de la sesión si no existe
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
if 'image' not in st.session_state:
    st.session_state.image = None
    
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
    
if 'result' not in st.session_state:
    st.session_state.result = None

# Cargar el modelo solo una vez
if not st.session_state.model_loaded:
    try:
        with st.spinner("Cargando modelo..."):
            model = load_model()
            st.session_state.model = model
            st.session_state.model_loaded = True
        st.success("¡Modelo cargado correctamente!")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.warning("Por favor, asegúrate de tener el archivo 'resnet_ecommerce_classifier.pth' en el mismo directorio")
        st.stop()
else:
    model = st.session_state.model

# Función para realizar análisis al hacer clic en el botón
def analyze_image():
    if st.session_state.image is not None:
        with st.spinner("Analizando imagen..."):
            # Realizar predicción
            try:
                # Simular carga con una barra de progreso
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)  # Reducido para mejor experiencia
                    progress_bar.progress(i)
                
                # Realizar predicción real
                st.session_state.result = predict(st.session_state.image, model)
                st.session_state.analyze_clicked = True
                
                # Eliminar la barra de progreso después de completar
                progress_bar.empty()
                st.success("¡Análisis completado!")
            except Exception as e:
                st.error(f"Error durante la predicción: {str(e)}")
                st.session_state.analyze_clicked = False
                st.session_state.result = None

# Dividir la pantalla en dos columnas
col1, col2 = st.columns(2)

# Primera columna: Subida y visualización de imagen
with col1:
    st.markdown("<h3>Sube tu imagen</h3>", unsafe_allow_html=True)
    
    # Subir una imagen
    uploaded_file = st.file_uploader("Selecciona una imagen de producto", type=["jpg", "jpeg", "png"], key="uploader")
    
    # Botón para analizar (aparece arriba)
    analyze_disabled = uploaded_file is None
    if st.button("🔍 Analizar Imagen", disabled=analyze_disabled, key="analyze_button"):
        analyze_image()
    
    # Mostrar la imagen si está cargada
    if uploaded_file is not None:
        try:
            # Guardar imagen en el estado de la sesión
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.image = image
            
            # Mostrar imagen
            st.image(image, caption='Imagen subida', use_column_width=True)
        except Exception as e:
            st.error(f"Error al cargar la imagen: {str(e)}")
            st.session_state.image = None
    else:
        # Resetear el estado si no hay imagen
        st.session_state.image = None
        st.session_state.analyze_clicked = False
        st.session_state.result = None
        st.info("👆 Sube una imagen para comenzar")

# Segunda columna: Resultados del análisis
with col2:
    st.markdown("<h3>Resultados del análisis</h3>", unsafe_allow_html=True)
    
    # Mostrar resultados solo si se ha analizado una imagen
    if st.session_state.image is None:
        st.markdown('<div class="empty-column">Carga una imagen para ver los resultados aquí</div>', unsafe_allow_html=True)
    elif not st.session_state.analyze_clicked or st.session_state.result is None:
        st.markdown('<div class="empty-column">Haz clic en "Analizar Imagen" para ver los resultados</div>', unsafe_allow_html=True)
    else:
        # Mostrar resultado con la categoría predicha
        result = st.session_state.result
        
        # Contenedor de resultados
        st.markdown(f"""
        <div class="result-box">
            <h2 style="margin-top: 0; color: #1E90FF;">Resultado del Análisis</h2>
            <h3>Categoría: <span style="color: #1E90FF; font-weight: bold;">{result['class'].upper()}</span></h3>
            <h4>Confianza: <span style="color: #1E90FF;">{result['confidence']:.2f}%</span></h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar gráfico de confianza para todas las clases
        st.subheader("Confianza por categoría")
        
        # Ordenar los valores de confianza de mayor a menor
        confidences = result['all_probabilities']
        sorted_confidences = {k: v for k, v in sorted(confidences.items(), key=lambda item: item[1], reverse=True)}
        
        # Crear barras coloridas para cada clase de forma separada (no anidada en el mismo contenedor)
        for cls, conf in sorted_confidences.items():
            # Color más intenso para la clase predicha
            color = "#1E90FF" if cls == result['class'] else "#A9A9A9"
            
            # Asegurar que el ancho sea visible incluso con valores pequeños
            width = max(conf, 1)
            
            # Contenedor separado para cada barra
            st.markdown(f"""
            <div class="confidence-bar-container">
                <span class="confidence-bar-label">{cls}</span>
                <div class="confidence-bar-outer">
                    <div class="confidence-bar-inner" style="width: {width}%; background-color: {color};">
                        <span class="confidence-bar-text">{conf:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 E-commerce Product Classifier | Powered by ResNet</div>", unsafe_allow_html=True)