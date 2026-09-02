# Прогнозирование целевых действий пользователей СберАвтоподписки

## 📋 Описание проекта

Машинное обучение проект, разработанный в рамках финальной работы курса **ML Engineer** от Skillbox. Проект направлен на предсказание вероятности совершения целевого действия (оставление заявки/заказ звонка) пользователями сайта «СберАвтоподписка» на основе их поведения во время визита.

**СберАвтоподписка** — сервис долгосрочной аренды автомобилей для физических лиц, альтернатива автокредиту и каршерингу.

## 🎯 Цели проекта

- Провести разведочный анализ данных (EDA) поведения пользователей
- Выявить ключевые факторы, влияющие на конверсию
- Построить ML-модель для прогнозирования целевых действий
- Развернуть модель в виде REST API для продакшн-использования

## 📊 Данные

Использованы данные из Google Analytics (last-click attribution model) по сайту «СберАвтоподписка»:

- **ga_sessions.pkl** — информация о визитах (1.86 млн записей)
- **ga_hits-002.pkl** — информация о событиях/хитах (15.7 млн записей)

### Основные признаки:
- `session_id` — идентификатор визита
- `visit_number` — порядковый номер визита клиента
- `utm_source`, `utm_medium`, `utm_campaign` — параметры рекламных кампаний
- `device_category`, `device_os`, `device_brand`, `device_browser` — информация об устройстве
- `geo_city`, `geo_country` — геолокация пользователя
- `hit_number` — количество событий в рамках визита

### Целевая переменная:
- `is_target` — факт совершения целевого действия (1 — совершено, 0 — не совершено)

Целевые действия включают: `sub_car_claim_click`, `sub_car_claim_submit_click`, `sub_open_dialog_click`, `sub_custom_question_submit_click`, `sub_call_number_click`, `sub_callback_submit_click`, `sub_submit_success`, `sub_car_request_submit_click`

### Технологии:

- **Python 3.12+**
- **Pandas, NumPy** — обработка данных
- **Scikit-learn** — машинное обучение
- **Matplotlib, Seaborn** — визуализация
- **FastAPI** — REST API для деплоя модели
- **Pydantic** — валидация данных
- **Dill** — сериализация модели

## 🚀 Установка и запуск

### 1. Клонируйте репозиторий и перейдите в папку проекта
```bash
git clone https://github.com/antonsv26-ops/Final-project-specialist-Machine-Learning-Engineer.git
cd Final-project-specialist-Machine-Learning-Engineer/"Финальная работа"/final
```

### 2. Установите зависимости
```bash
pip install -r requirements.txt
```

### 3. Запустите API сервер
```bash
uvicorn main:app --reload
```
Сервер запустится локально по адресу: http://localhost:8000

## Тестирование API

Откройте документацию Swagger UI: http://localhost:8000/docs
или отправьте POST запрос:

    curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "session_id": "test_session_123",
           "visit_number": 1,
           "hit_number": 15,
           "utm_source": "google",
           "utm_medium": "cpc",
           "utm_campaign": "test_campaign",
           "utm_keyword": "автоподписка",
           "device_category": "mobile",
           "device_os": "Android",
           "device_brand": "Samsung",
           "device_browser": "Chrome",
           "device_model": "Galaxy S21",
           "device_screen_resolution": "1080x2400",
           "geo_city": "Moscow",
           "geo_country": "Russia"
         }'

Пример ответа:
json
{
  "session_id": "test_session_123",
  "Result": 0.73
}

📈 Результаты

Модель: RandomForestClassifier

ROC-AUC: 0.8524


‍💻 Автор

Антон Савинов
