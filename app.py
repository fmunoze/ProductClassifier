from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import io
import base64
import os

app = Flask(__name__)

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

# Variable global para mantener el modelo cargado
model = None

# Función para cargar el modelo
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
        probabilities = F.softmax(outputs, dim=1)[0] * 100
        
    return {
        'class': classes[predicted.item()],
        'confidence': probabilities[predicted.item()].item(),
        'all_probabilities': {classes[i]: probabilities[i].item() for i in range(len(classes))}
    }

# Cargar el modelo al iniciar
print("Cargando modelo...")
try:
    model = load_model()
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de tener el archivo 'resnet_ecommerce_classifier.pth' en el mismo directorio")

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para procesar la imagen
@app.route('/predict', methods=['POST'])
def process_image():
    global model
    
    if model is None:
        try:
            model = load_model()
            print("Modelo cargado correctamente")
        except Exception as e:
            return jsonify({'error': f'No se pudo cargar el modelo: {str(e)}'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha subido ninguna imagen'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
    
    try:
        # Leer y procesar la imagen
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Codificar imagen para mostrarla en el cliente
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Realizar predicción
        result = predict(img, model)
        
        # Ordenar probabilidades
        sorted_probs = sorted(
            result['all_probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Formatear resultados para la respuesta
        return jsonify({
            'success': True,
            'predicted_class': result['class'],
            'confidence': result['confidence'],
            'probabilities': sorted_probs,
            'image': img_str
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Crear los directorios necesarios para las plantillas y los archivos estáticos
if not os.path.exists('templates'):
    os.makedirs('templates')
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/css'):
    os.makedirs('static/css')
if not os.path.exists('static/js'):
    os.makedirs('static/js')

# Escribir el archivo HTML
with open('templates/index.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clasificador de Productos</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Clasificador Avanzado de Productos</h1>
            <p class="subtitle">Powered by ResNet: Sube una imagen para clasificar el producto</p>
        </header>

        <div class="info-container">
            <div class="info-button" onclick="toggleInfo()">
                <i class="fas fa-info-circle"></i> Acerca de esta app
            </div>
            <div class="info-content" id="infoContent">
                <p>Esta aplicación utiliza un modelo de deep learning ResNet50 para clasificar imágenes de productos en cuatro categorías:</p>
                <ul>
                    <li>👖 Jeans</li>
                    <li>🛋️ Sofás</li>
                    <li>👕 Camisetas</li>
                    <li>📺 Televisores</li>
                </ul>
                <p>El modelo ha sido entrenado con técnicas avanzadas de transfer learning para ofrecer mayor precisión.</p>
                <h4>¿Cómo usar?</h4>
                <ol>
                    <li>Sube una imagen de un producto</li>
                    <li>Haz clic en "Analizar Imagen"</li>
                    <li>Observa los resultados</li>
                </ol>
            </div>
        </div>

        <main>
            <div class="columns">
                <div class="column">
                    <h3>Sube tu imagen</h3>
                    <div class="upload-area" id="uploadArea">
                        <input type="file" id="fileInput" accept="image/*" hidden>
                        <div class="upload-prompt" id="uploadPrompt">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <p>Arrastra una imagen aquí o haz clic para seleccionar</p>
                        </div>
                        <div class="image-preview" id="imagePreview" hidden>
                            <img id="previewImg" src="" alt="Preview">
                        </div>
                    </div>
                    <button class="analyze-btn" id="analyzeBtn" disabled>
                        <i class="fas fa-search"></i> Analizar Imagen
                    </button>
                    <div class="loading" id="loading" hidden>
                        <div class="spinner"></div>
                        <p>Analizando imagen...</p>
                    </div>
                </div>
                <div class="column">
                    <h3>Resultados del análisis</h3>
                    <div class="results-container" id="resultsContainer">
                        <div class="empty-results" id="emptyResults">
                            <i class="fas fa-image"></i>
                            <p>Carga una imagen para ver los resultados aquí</p>
                        </div>
                        <div class="results" id="results" hidden>
                            <div class="result-box">
                                <h2>Resultado del Análisis</h2>
                                <h3>Categoría: <span id="predictedClass"></span></h3>
                                <h4>Confianza: <span id="confidence"></span></h4>
                            </div>
                            <h3>Confianza por categoría</h3>
                            <div id="confidenceBars"></div>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <footer>
            <p>© 2025 E-commerce Product Classifier | Powered by ResNet</p>
        </footer>
    </div>

    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>''')

# Escribir el archivo CSS
with open('static/css/style.css', 'w') as f:
    f.write('''/* Estilos generales */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

body {
    background-color: #f8f9fa;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Encabezado */
header {
    text-align: center;
    margin-bottom: 30px;
}

h1 {
    color: #1E90FF;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.subtitle {
    color: #555;
    font-size: 1.2rem;
}

/* Sección de información */
.info-container {
    margin-bottom: 30px;
}

.info-button {
    display: inline-block;
    background-color: #f0f8ff;
    color: #1E90FF;
    padding: 10px 15px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
}

.info-button:hover {
    background-color: #1E90FF;
    color: white;
}

.info-content {
    display: none;
    background-color: #f0f8ff;
    border-left: 5px solid #1E90FF;
    padding: 15px;
    margin-top: 10px;
    border-radius: 5px;
}

.info-content ul, .info-content ol {
    margin-left: 20px;
    margin-bottom: 10px;
}

/* Columnas */
.columns {
    display: flex;
    gap: 20px;
}

.column {
    flex: 1;
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

h3 {
    color: #1E90FF;
    font-size: 1.3rem;
    margin-bottom: 20px;
}

/* Área de carga de imágenes */
.upload-area {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
    min-height: 250px;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    transition: all 0.3s;
}

.upload-area:hover {
    border-color: #1E90FF;
    background-color: #f0f8ff;
}

.upload-prompt {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.upload-prompt i {
    font-size: 3rem;
    color: #ccc;
    margin-bottom: 10px;
}

.image-preview {
    width: 100%;
    display: flex;
    justify-content: center;
}

.image-preview img {
    max-width: 100%;
    max-height: 300px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Botón de análisis */
.analyze-btn {
    background-color: #1E90FF;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 12px;
    width: 100%;
    font-size: 1rem;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s;
}

.analyze-btn:hover:not(:disabled) {
    background-color: #0078FF;
}

.analyze-btn:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}

/* Cargando */
.loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 20px;
}

.spinner {
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-left-color: #1E90FF;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin-bottom: 10px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Resultados */
.results-container {
    min-height: 300px;
}

.empty-results {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    background-color: #f8f9fa;
    border-radius: 10px;
    border: 2px dashed #ccc;
    color: #666;
    text-align: center;
    padding: 20px;
}

.empty-results i {
    font-size: 3rem;
    color: #ccc;
    margin-bottom: 10px;
}

.result-box {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    border-left: 5px solid #1E90FF;
}

.result-box h2 {
    margin-top: 0;
    color: #1E90FF;
    font-size: 1.8rem;
    margin-bottom: 15px;
}

.result-box h3 {
    color: #333;
    margin-bottom: 10px;
}

.result-box span {
    color: #1E90FF;
    font-weight: bold;
}

/* Barras de confianza */
.confidence-bar {
    margin-bottom: 15px;
}

.confidence-bar-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
}

.confidence-bar-label {
    font-weight: bold;
}

.confidence-bar-outer {
    width: 100%;
    background-color: #f0f0f0;
    border-radius: 5px;
    height: 25px;
    overflow: hidden;
}

.confidence-bar-inner {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    color: white;
    font-weight: bold;
    transition: width 1s ease-in-out;
}

/* Footer */
footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
    color: #666;
    font-size: 0.9rem;
}

/* Responsive */
@media (max-width: 768px) {
    .columns {
        flex-direction: column;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    .subtitle {
        font-size: 1rem;
    }
}''')

# Escribir el archivo JavaScript
with open('static/js/script.js', 'w') as f:
    f.write('''// Variables para los elementos DOM
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const emptyResults = document.getElementById('emptyResults');
const results = document.getElementById('results');
const predictedClass = document.getElementById('predictedClass');
const confidence = document.getElementById('confidence');
const confidenceBars = document.getElementById('confidenceBars');
const infoContent = document.getElementById('infoContent');

// Event listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
analyzeBtn.addEventListener('click', analyzeImage);

// Función para mostrar/ocultar información
function toggleInfo() {
    if (infoContent.style.display === 'block') {
        infoContent.style.display = 'none';
    } else {
        infoContent.style.display = 'block';
    }
}

// Función para manejar la selección de archivo
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        displayImage(file);
    }
}

// Función para manejar el arrastre sobre el área
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

// Función para manejar cuando el arrastre sale del área
function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

// Función para manejar la soltura del archivo
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files;
        displayImage(file);
    }
}

// Función para mostrar la imagen seleccionada
function displayImage(file) {
    if (!file.type.match('image.*')) {
        alert('Por favor, selecciona una imagen válida');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        uploadPrompt.hidden = true;
        imagePreview.hidden = false;
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Función para analizar la imagen
function analyzeImage() {
    if (!fileInput.files[0]) {
        alert('Por favor, selecciona una imagen primero');
        return;
    }

    // Mostrar cargando y ocultar otros elementos
    loading.hidden = false;
    analyzeBtn.disabled = true;
    emptyResults.hidden = true;
    results.hidden = true;

    // Crear FormData para enviar el archivo
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Enviar la solicitud al servidor
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Ha ocurrido un error al procesar la imagen');
    })
    .finally(() => {
        loading.hidden = true;
        analyzeBtn.disabled = false;
    });
}

// Función para mostrar los resultados
function displayResults(data) {
    // Mostrar la clase predicha y la confianza
    predictedClass.textContent = data.predicted_class.toUpperCase();
    confidence.textContent = data.confidence.toFixed(2) + '%';

    // Limpiar y crear las barras de confianza
    confidenceBars.innerHTML = '';
    data.probabilities.forEach(item => {
        const [cls, conf] = item;
        const isHighest = cls === data.predicted_class;
        const color = isHighest ? '#1E90FF' : '#A9A9A9';

        const barContainer = document.createElement('div');
        barContainer.className = 'confidence-bar';

        const barHeader = document.createElement('div');
        barHeader.className = 'confidence-bar-header';

        const label = document.createElement('span');
        label.className = 'confidence-bar-label';
        label.textContent = cls;

        const percentage = document.createElement('span');
        percentage.textContent = conf.toFixed(1) + '%';

        barHeader.appendChild(label);
        barHeader.appendChild(percentage);

        const outerBar = document.createElement('div');
        outerBar.className = 'confidence-bar-outer';

        const innerBar = document.createElement('div');
        innerBar.className = 'confidence-bar-inner';
        innerBar.style.backgroundColor = color;
        innerBar.style.width = '0%'; // Comenzar en 0 para animación

        // Añadir texto en la barra
        const innerText = document.createElement('span');
        innerText.textContent = conf.toFixed(1) + '%';
        innerBar.appendChild(innerText);

        outerBar.appendChild(innerBar);
        
        barContainer.appendChild(barHeader);
        barContainer.appendChild(outerBar);
        
        confidenceBars.appendChild(barContainer);

        // Animar la barra
        setTimeout(() => {
            innerBar.style.width = Math.max(conf, 1) + '%';
        }, 100);
    });

    // Mostrar los resultados
    emptyResults.hidden = true;
    results.hidden = false;
}''')

# Código principal para ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)