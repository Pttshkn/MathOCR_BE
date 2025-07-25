import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from formula_recognizer import FormulaRecognizer

# Конфигурация
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'crnn_model.pth')

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Ограничение 5MB

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# Ленивая инициализация модели
recognizer = None

@app.route('/')
def home():
    return "MathOCR API is running!"

@app.route('/recognize', methods=['POST'])
def recognize_formula():
    global recognizer
    if not recognizer:
        recognizer = FormulaRecognizer(MODEL_PATH)
    
    if 'file' not in request.files:
        return jsonify({"error": "Файл не найден в запросе"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Не выбран файл"}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({"error": "Неподдерживаемый формат файла"}), 400

    try:
        # Чтение изображения в память
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Распознавание
        latex_result = recognizer.recognize(img)
        
        return jsonify({
            "result": latex_result,
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({
            "error": f"Ошибка обработки: {str(e)}",
            "type": type(e).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "active",
        "device": str(recognizer.DEVICE) if recognizer else "Not initialized"
    })
