import streamlit as st
import pandas as pd
import joblib
import time

# --- НАСТРОЙКА СТРАНИЦЫ (Вкладка в браузере) ---
st.set_page_config(page_title="Astana Weather AI", page_icon="🌤️", layout="centered")

# --- ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ И СКАЛЕРА ---
@st.cache_resource
def load_resources():
    # Загружаем файлы .pkl, которые мы сохранили в Jupyter
    model = joblib.load('weather_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    # Пытаемся загрузить обученные ресурсы
    model, scaler = load_resources()
    
    # --- ГЛАВНЫЙ ЗАГОЛОВОК ИНТЕРФЕЙСА (На английском) ---
    st.title("🌤️ Astana Temperature Predictor")
    st.caption("Advanced Machine Learning Pipeline powered by NASA POWER Data")
    st.write("Adjust the meteorological parameters on the left to simulate and predict Astana's ambient temperature.")
    st.markdown("---")

    # --- БОКОВАЯ ПАНЕЛЬ / ПАНЕЛЬ УПРАВЛЕНИЯ (На английском) ---
    st.sidebar.header("🕹️ Control Panel")
    st.sidebar.markdown("Tune the atmospheric parameters:")
    
    # Слайдеры для ввода данных пользователем
    humidity = st.sidebar.slider('Relative Humidity (%)', 10, 100, 55)
    pressure = st.sidebar.slider('Surface Pressure (kPa)', 90, 110, 101)
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', 0.0, 25.0, 6.5)

    # Вычисляем наш кастомный признак взаимодействия (Feature Engineering)
    weather_index = humidity * wind_speed

    # --- ЖИВОЙ МОНИТОР ДАННЫХ / КАРТОЧКИ (На английском) ---
    st.subheader("📊 Live Input Monitor")
    col1, col2, col3 = st.columns(3)
    col1.metric("Humidity", f"{humidity} %", "💧")
    col2.metric("Pressure", f"{pressure} kPa", "🧭")
    col3.metric("Wind Speed", f"{wind_speed} m/s", "💨")

    # Формируем таблицу DataFrame для передачи в модель (структура как при обучении)
    input_data = pd.DataFrame([[humidity, pressure, wind_speed, weather_index]], 
                              columns=['humidity', 'pressure', 'wind_speed', 'weather_index'])

    st.markdown("---")

    # --- КНОПКА ЗАПУСКА И ПРЕДСКАЗАНИЕ ---
    if st.sidebar.button('🚀 Run AI Prediction'):
        # Анимация загрузки (эффект "думающего" компьютера)
        with st.spinner('Model is processing features...'):
            time.sleep(0.5) 
            
            # Масштабируем введенные данные и делаем прогноз
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
        
        # Вывод результата работы ИИ (На английском)
        st.subheader("🎯 Model Output")
        st.success(f"### Predicted Temperature: {prediction:.2f} °C")
        
        # ДИНАМИЧЕСКИЕ ЭФФЕКТЫ И ПРЕДУПРЕЖДЕНИЯ (В зависимости от прогноза)
        if prediction <= -15:
            st.snow() # Включает падение снега на экране, если мороз ниже -15
            st.info("🥶 **Extreme Frost Alert:** Absolute classic for Astana winter! Bundle up or stay inside.")
        elif -15 < prediction <= 0:
            st.info("❄️ **Freezing Conditions:** Typical cold weather. Watch out for black ice!")
        elif 0 < prediction <= 15:
            st.warning("🍂 **Cool/Brisk Air:** Perfect jacket weather. Standard spring/autumn vibe.")
        elif 15 < prediction <= 28:
            st.success("☀️ **Pleasant & Warm:** Excellent weather! Enjoy the day in the capital.")
        else:
            st.error("🔥 **Unusual Heatwave:** Exceptionally warm for this region. Stay hydrated!")

    # --- ИНТЕРАКТИВНЫЙ БЛОК С ФАКТАМИ (На английском) ---
    st.markdown("---")
    with st.expander("💡 Did you know about Astana's climate?"):
        st.write("""
        * **Extreme Range:** Astana is the second-coldest capital city in the world, with temperatures sometimes dropping below -40°C in winter and rising above +35°C in summer!
        * **Wind Factor:** The interaction between humidity and high wind speed (your `weather_index` feature!) drastically changes how cold the temperature actually feels.
        """)
            
except FileNotFoundError:
    # Обработка ошибки, если файлы моделей забыли положить в папку
    st.sidebar.error("⚠️ Error: Missing trained files.")
    st.error("Please ensure 'weather_model.pkl' and 'scaler.pkl' are uploaded to the repository.")