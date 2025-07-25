import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from tqdm import tqdm
import io

# Абсолютные пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "crnn_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FormulaRecognizer:
    def __init__(self, model_path):
        # Проверка существования файла
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.char_set = checkpoint['char_set']
        self.model = CRNN(len(self.char_set)).to(DEVICE)
        self.model.load_state_dict(checkpoint['state_dict'])  # Исправлено
        self.model.eval()
        self.DEVICE = DEVICE
        
        # Трансформы для распознавания
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def recognize(self, img_array):
        try:
            # Конвертация в grayscale
            if len(img_array.shape) == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                img = img_array
            
            # Бинаризация
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Применение трансформов
            img_tensor = self.transform(binary).unsqueeze(0).to(DEVICE)
            
            # Подача в модель
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = outputs.log_softmax(2).exp()
                _, preds = torch.max(probs, 2)
                preds = preds.squeeze(1).cpu().numpy()
                
                # Декодирование
                pred_str = ''.join([self.char_set[i] for i in preds if i != len(self.char_set)-1])
                pred_str = ''.join([c for i, c in enumerate(pred_str) if i == 0 or c != pred_str[i-1]])
            
            return pred_str if pred_str else "<Пустое предсказание>"
        except Exception as e:
            return f"Ошибка распознавания: {str(e)}"

class CRNN(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        # Упрощенная CNN архитектура
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        
        # RNN с dropout
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(512, num_chars)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Инициализация forget gate
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN
        x = self.cnn(x)
        
        # Подготовка для RNN
        x = x.squeeze(2)  # [batch, channels, width]
        x = x.permute(2, 0, 1)  # [width, batch, channels]
        
        # RNN
        x, _ = self.rnn(x)
        
        # FC с ограничением значений
        x = self.fc(x)
        return x

class FormulaDataset(Dataset):
    def __init__(self, data, char_set):
        self.char_set = char_set
        self.char_to_idx = {c:i for i,c in enumerate(char_set)}
        self.data = []
        
        # Фильтрация данных
        for img, label in data:
            try:
                # Проверка изображения
                if img.size == 0:
                    raise ValueError("Пустое изображение")
                
                # Проверка метки
                if not label:
                    raise ValueError("Пустая метка")
                
                # Проверка символов
                invalid_chars = [c for c in label if c not in self.char_to_idx]
                if invalid_chars:
                    raise ValueError(f"Неизвестные символы: {invalid_chars}")
                
                self.data.append((img, label))
            except Exception as e:
                print(f"Пропуск элемента: {str(e)}")
        
        # Трансформы с аугментацией
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 256)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return self.transform(img), target

def collate_fn(batch):
    """Кастомная функция для объединения данных в батчи"""
    images = torch.stack([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    target_lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    return images, targets, target_lengths

def train():
    # Загрузка данных
    print("\nЗагрузка данных...")
    train_data = load_dataset(os.path.join(DATASET_ROOT, "train"))
    val_data = load_dataset(os.path.join(DATASET_ROOT, "val"))
    
    if not train_data:
        raise RuntimeError("Не найдено тренировочных данных")
    if not val_data:
        print("Предупреждение: не найдено валидационных данных")
    
    # Создание словаря символов
    char_set = sorted(set("".join([label for _,label in train_data+val_data]))) + [' ']
    print(f"\nСловарь символов ({len(char_set)}): {char_set}")
    
    # Датасеты
    train_dataset = FormulaDataset(train_data, char_set)
    val_dataset = FormulaDataset(val_data, char_set) if val_data else None
    
    # Проверка данных
    print("\nПроверка данных:")
    print(f"Тренировочных примеров: {len(train_dataset)}")
    if val_dataset:
        print(f"Валидационных примеров: {len(val_dataset)}")
    
    sample_img, sample_label = train_dataset[0]
    print(f"\nПример данных:")
    print(f"Размер изображения: {sample_img.shape}")
    print(f"Метка: {sample_label.tolist()} -> '{''.join([char_set[i] for i in sample_label])}'")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                            collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, 
                          collate_fn=collate_fn, num_workers=2) if val_dataset else None
    
    # Модель
    model = CRNN(len(char_set)).to(DEVICE)
    
    # Оптимизатор и планировщик
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    
    # Функция потерь
    criterion = nn.CTCLoss(blank=len(char_set)-1, zero_infinity=True)
    
    # Обучение
    print("\nНачало обучения...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/100 [Train]')
        
        for images, targets, target_lengths in progress_bar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Подготовка для CTC
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=DEVICE
            )
            
            # Вычисление потерь
            loss = criterion(
                outputs.log_softmax(2),  # Важно: log_softmax вместо raw выходов
                targets,
                input_lengths,
                target_lengths
            )
            
            # Проверка на NaN
            if torch.isnan(loss):
                print("\nОбнаружен NaN в loss, пропуск батча")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Клиппинг градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Валидация
        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for images, targets, target_lengths in tqdm(val_loader, desc=f'Epoch {epoch+1}/20 [Val]'):
                    images = images.to(DEVICE)
                    targets = targets.to(DEVICE)
                    
                    outputs = model(images)
                    input_lengths = torch.full(
                        (images.size(0),),
                        outputs.size(0),
                        dtype=torch.long,
                        device=DEVICE
                    )
                    
                    loss = criterion(
                        outputs.log_softmax(2),
                        targets,
                        input_lengths,
                        target_lengths
                    )
                    val_loss += loss.item()
        
        # Логирование
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if val_loader else 0
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        if val_loader:
            print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Пример предсказания
        sample_img, sample_label = train_dataset[0]
        with torch.no_grad():
            model.eval()
            outputs = model(sample_img.unsqueeze(0).to(DEVICE))
            probs = outputs.log_softmax(2).exp()
            _, preds = torch.max(probs, 2)
            pred_str = ''.join([char_set[i] for i in preds[0].cpu().numpy() 
                              if i != len(char_set)-1])
            pred_str = ''.join([c for i,c in enumerate(pred_str) 
                              if i==0 or c!=pred_str[i-1]])
        
        print(f"\nПример предсказания:")
        print(f"Оригинал: '{''.join([char_set[i] for i in sample_label])}'")
        print(f"Предсказание: '{pred_str}'")
        
        # Сохранение модели
        torch.save({
            'state_dict': model.state_dict(),
            'char_set': char_set,
            'epoch': epoch+1,
            'val_loss': avg_val_loss
        }, MODEL_PATH)
        
        # Обновление learning rate
        if val_loader:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step(avg_train_loss)

class FormulaRecognizer:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.char_set = checkpoint['char_set']
        self.model = CRNN(len(self.char_set)).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

    def recognize(self, img_array):
        try:
            # Конвертация цветного изображения в grayscale
            if len(img_array.shape) == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                img = img_array
            
            # Бинаризация и нормализация
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img_tensor = torch.from_numpy(binary.astype(np.float32) / 255.0)
            img_tensor = (img_tensor.unsqueeze(0).unsqueeze(0) - 0.5) / 0.5
            
            # Подаем в модель
            with torch.no_grad():
                outputs = self.model(img_tensor.to(DEVICE))
                probs = outputs.log_softmax(2).exp()
                _, preds = torch.max(probs, 2)
                pred_str = ''.join([self.char_set[i] for i in preds[0].cpu().numpy() 
                                if i != len(self.char_set)-1])
                pred_str = ''.join([c for i,c in enumerate(pred_str) 
                                if i==0 or c!=pred_str[i-1]])
            
            return pred_str if pred_str else "<Пустое предсказание>"
        except Exception as e:
            print(f"Ошибка распознавания: {str(e)}")
            return f"Ошибка: {str(e)}"

# Инициализация распознавателя при запуске
recognizer = FormulaRecognizer(MODEL_PATH)

def allowed_file(filename):
    """Проверка допустимого расширения файла"""
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/recognize', methods=['POST'])
def handle_recognize():
    """Обработка запроса на распознавание"""
    # Проверка наличия файла в запросе
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не предоставлен'}), 400
    
    file = request.files['file']
    
    # Проверка имени файла
    if file.filename == '':
        return jsonify({'error': 'Не выбрано изображение'}), 400
    
    # Проверка формата файла
    if not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый формат файла. Разрешены: png, jpg, jpeg'}), 400
    
    try:
        # Чтение изображения в память
        img_bytes = file.read()
        
        # Конвертация в numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Распознавание формулы
        result = recognizer.recognize(img)
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервера"""
    return jsonify({'status': 'active', 'device': str(DEVICE)})

if __name__ == "__main__":
    # Режим работы: обучение или запуск сервера
    MODE = "recognize"  # "train" или "recognize"
    
    if MODE == "train":
        print("Запуск обучения модели...")
        train()
    elif MODE == "recognize":
        print(f"Запуск сервера распознавания формул на устройстве: {DEVICE}")
        app.run(host='0.0.0.0', port=5000)
