// ======= Переключение темы =======
const themeToggle = document.getElementById('theme-toggle');
const body = document.body;

// Проверка сохранённой темы
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    body.setAttribute('data-theme', savedTheme);
    themeToggle.checked = savedTheme === 'dark';
}

// Обработчик переключателя
themeToggle.addEventListener('change', () => {
    const theme = themeToggle.checked ? 'dark' : 'light';
    body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
});

// ======= Загрузка файла =======
const fileUpload = document.getElementById('file-upload');
const preview = document.getElementById('preview');
const dropZone = document.getElementById('drop-zone');
const recognizeBtn = document.getElementById('recognize-btn');
const imageContainer = document.querySelector('.image-container');
const errorMessage = document.createElement('div');
errorMessage.className = 'error-message';
dropZone.after(errorMessage);

// Удаление изображения
document.querySelector('.remove-image-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    preview.src = '';
    imageContainer.style.display = 'none';
    fileUpload.value = '';
    errorMessage.style.display = 'none';
    recognizeBtn.disabled = false;
});

// Обработчик выбора файла
fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

// Функция обработки файла
function handleFile(file) {
    errorMessage.style.display = 'none';
    recognizeBtn.disabled = false;
    
    if (!file) {
        imageContainer.style.display = 'none';
        return;
    }

    // Проверка формата
    if (!file.type.match('image.*')) {
        errorMessage.textContent = '❌ Поддерживаются только изображения (JPEG, PNG, GIF)';
        errorMessage.style.display = 'block';
        recognizeBtn.disabled = true;
        imageContainer.style.display = 'none';
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        preview.src = e.target.result;
        imageContainer.style.display = 'inline-block';
    };
    reader.readAsDataURL(file);
}

// ======= Drag and Drop =======
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropZone.classList.add('drag-over');
}

function unhighlight() {
    dropZone.classList.remove('drag-over');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file) {
        fileUpload.files = dt.files;
        handleFile(file);
    }
}

// ======= Вставка изображения из буфера (Ctrl+V) =======
document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            const file = item.getAsFile();
            if (file) {
                // Помещаем файл в <input type="file">
                const dt = new DataTransfer();
                dt.items.add(file);
                fileUpload.files = dt.files;
                // Отображаем превью и активируем кнопку
                handleFile(file);
                // Подсветка зоны вставки
                dropZone.classList.add('drag-over');
                setTimeout(() => dropZone.classList.remove('drag-over'), 500);
            }
            break;
        }
    }
});

// ======= Копирование результата =======
document.getElementById('copy-btn').addEventListener('click', function() {
    const btn = this;
    const latex = document.getElementById('result').textContent.trim();
    
    navigator.clipboard.writeText(latex).then(() => {
        btn.classList.add('copied-state');
        btn.innerHTML = '<i class="fas fa-check"></i><span>Скопировано!</span>';
        
        setTimeout(() => {
            btn.classList.remove('copied-state');
            btn.innerHTML = '<i class="far fa-copy"></i><span>Копировать LaTeX</span>';
        }, 2000);
    });
});

// ======= Распознавание формулы =======
document.getElementById('recognize-btn').addEventListener('click', async function() {
    if (!preview.src || imageContainer.style.display === 'none') {
        errorMessage.textContent = '❌ Пожалуйста, загрузите изображение';
        errorMessage.style.display = 'block';
        return;
    }
    
    if (recognizeBtn.disabled) {
        return;
    }

    // Показываем лоадер
    this.innerHTML = '<div class="loader"></div> Обработка...';
    this.disabled = true;
    errorMessage.style.display = 'none';  // Скрываем предыдущие ошибки

    try {
        // Создаем форму для отправки
        const formData = new FormData();
        formData.append('file', fileUpload.files[0]);
        
        // Отправляем на сервер
        const response = await fetch('https://mathocr-backend.onrender.com/recognize', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Показываем результат
        document.getElementById('result').textContent = data.result;
        
        // Обновляем MathJax для отображения формулы
        MathJax.typeset();
    } catch (error) {
        errorMessage.textContent = '❌ ' + error.message;
        errorMessage.style.display = 'block';
    } finally {
        this.innerHTML = '<i class="fas fa-robot"></i> Распознать';
        this.disabled = false;
    }
});
