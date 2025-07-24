import os
from flask import Flask, request, jsonify
from formula_recognizer import FormulaRecognizer

app = Flask(__name__)

# Инициализация модели при запуске
recognizer = FormulaRecognizer("model/crnn_model.pth")

@app.route('/')
def home():
    return "MathOCR API is running!"

@app.route('/recognize', methods=['POST'])
def recognize_formula():
    # Проверка наличия файла
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Сохранение временного файла
    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    try:
        # Распознавание формулы
        latex_result = recognizer.recognize(temp_path)
        return jsonify({"result": latex_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Удаление временного файла
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)