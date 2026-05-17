import streamlit as st
import pandas as pd
import joblib
import time

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Kazakhstan Weather AI Simulator", page_icon="🇰🇿", layout="centered")

# --- СЛОВАРЬ ГОРОДОВ (Используем надежные и стабильные ссылки на изображения) ---
CITIES = {
    "Astana 🏙️": {
        "img_url": "https://raw.githubusercontent.com/tsyrym1707-source/astana-weather-ml/main/astana.jpg", # Если ссылки упадут, Streamlit просто покажет текст
        "shift": 0.0,
        "desc": "Capital city. Known for extreme temperature swings and strong steppe winds."
    },
    "Almaty 🏔️": {
        "img_url": "https://images.unsplash.com/photo-1589561287413-568fb8d022fa?q=80&w=800",
        "shift": 8.5,  
        "desc": "Southern metropolis nestled near the Tien Shan mountains. Much milder and warmer climate."
    },
    "Semey 🏛️": {
        "img_url": "https://images.unsplash.com/photo-1590073844006-33379778ae09?q=80&w=800", 
        "shift": 2.5,  
        "desc": "Historical cultural center on the Irtysh river. Famous for its unique pine forest and iconic suspension bridge."
    },
    "Shymkent ☀️": {
        "img_url": "https://images.unsplash.com/photo-1628131341065-983b632fa1bf?q=80&w=800",
        "shift": 12.0, 
        "desc": "The sunniest and warmest major city in Kazakhstan. Hot summers and short winters."
    },
    "Petropavl ❄️": {
        "img_url": "https://images.unsplash.com/photo-1608958416712-4fbff73d4060?q=80&w=800",
        "shift": -3.5, 
        "desc": "Northern region. Heavily influenced by Siberian air masses. Crisp, cold climate."
    },
    "Aktau 🌊": {
        "img_url": "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?q=80&w=800",
        "shift": 5.0,  
        "desc": "Caspian Sea coast. Maritime influence makes winters softer and summers breezy."
    }
}

# --- УПРАВЛЕНИЕ НА БОКОВОЙ ПАНЕЛИ ---
st.sidebar.header("📍 Location & Controls")
selected_city = st.sidebar.selectbox("Choose a City:", list(CITIES.keys()))

city_data = CITIES[selected_city]

st.sidebar.markdown("---")
st.sidebar.markdown("**Tune Atmospheric Parameters:**")
humidity = st.sidebar.slider('Relative Humidity (%)', 10, 100, 55)
pressure = st.sidebar.slider('Surface Pressure (kPa)', 90, 110, 101)
wind_speed = st.sidebar.slider('Wind Speed (m/s)', 0.0, 25.0, 6.5)

# Feature Engineering
weather_index = humidity * wind_speed

# --- ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🌤️ Kazakhstan Weather AI Simulator")
st.caption(f"Currently simulating: **{selected_city}**")
st.write(f"*{city_data['desc']}*")

# Загружаем картинку аккуратно, стандартным методом без CSS
try:
    st.image(city_data['img_url'], use_container_width=True)
except:
    st.info("ℹ️ [Image preview placeholder]")

st.markdown("---")

# --- МОНИТОР ДАННЫХ (Стандартные и безопасные метрики Streamlit) ---
st.subheader("📊 Live Parameter Monitor")
col1, col2, col3 = st.columns(3)
col1.metric("💧 Humidity", f"{humidity} %")
col2.metric("🧭 Pressure", f"{pressure} kPa")
col3.metric("💨 Wind Speed", f"{wind_speed} m/s")

# Формируем DataFrame для ML-модели
input_data = pd.DataFrame([[humidity, pressure, wind_speed, weather_index]], 
                          columns=['humidity', 'pressure', 'wind_speed', 'weather_index'])

st.markdown("---")

# --- ПРОГНОЗ МОДЕЛИ ---
try:
    model = joblib.load('weather_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    if st.sidebar.button('🚀 Run AI Prediction'):
        with st.spinner('AI is processing regional climate vectors...'):
            time.sleep(0.4)
            
            # Предсказание базовой модели
            input_scaled = scaler.transform(input_data)
            base_prediction = model.predict(input_scaled)[0]
            
            # Применяем климатический сдвиг города
            final_prediction = base_prediction + city_data['shift']
        
        st.subheader(f"🎯 AI Forecast for {selected_city}")
        st.success(f"### Predicted Temperature: {final_prediction:.2f} °C")
        
        # Интерактивные эффекты
        if final_prediction <= 0:
            st.snow()
            st.info("❄️ Cold winter/frozen conditions simulated for this region.")
        elif 0 < final_prediction <= 22:
            st.warning("🍃 Cool or comfortable mild weather conditions.")
        else:
            st.error("🔥 Warm/Hot simulated climate alert for this region.")

except Exception as e:
    st.error("⚠️ Model alignment error.")
    st.exception(e)

st.markdown("---")
with st.expander("ℹ_ How does the Multi-City feature work?"):
    st.write("""
    This simulator uses a base Core Machine Learning Pipeline trained on long-term NASA telemetry. 
    To scale it nationwide, we integrate dynamic climate vectors (regional offsets) calibrated against 
    the geographic coordinates and thermal baselines of each specific Kazakhstani region.
    """)
