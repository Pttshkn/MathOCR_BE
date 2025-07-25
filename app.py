import os
import uuid
from flask import Flask, request, jsonify
from formula_recognizer import FormulaRecognizer

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Ограничение 5MB

# Инициализация модели
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'crnn_model.pth')
recognizer = FormulaRecognizer("model/crnn_model.pth")

@app.route('/')
def home():
    return "MathOCR API is running!"

@app.route('/recognize', methods=['POST'])
def recognize_formula():
    # Проверка наличия файла
    if 'file' not in request.files:
        return jsonify({"error": "Файл не найден в запросе"}), 400
    
    file = request.files['file']
    
    # Проверка имени файла
    if file.filename == '':
        return jsonify({"error": "Не выбран файл"}), 400
    
    # Проверка расширения файла
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({"error": "Неподдерживаемый формат файла"}), 400

    try:
        # Генерируем уникальное имя файла
        temp_filename = f"temp_{uuid.uuid4().hex}.png"
        temp_path = os.path.join('/tmp', temp_filename)
        
        # Сохраняем файл
        file.save(temp_path)
        
        # Распознаем формулу
        latex_result = recognizer.recognize(temp_path)
        
        return jsonify({
            "result": latex_result,
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({
            "error": f"Ошибка обработки: {str(e)}",
            "type": type(e).__name__
        }), 500
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)
