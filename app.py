import streamlit as st
import pandas as pd
import joblib
import time

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Kazakhstan Weather AI Simulator", page_icon="🇰🇿", layout="centered")

# --- СЛОВАРЬ ГОРОДОВ: ФОНЫ И КЛИМАТИЧЕСКИЕ КОРРЕКТИРОВКИ ---
CITIES = {
    "Astana 🏙️": {
        "bg_url": "https://images.unsplash.com/photo-1578318182933-28952f4eb27b?q=80&w=1200",
        "shift": 0.0,
        "desc": "Capital city. Known for extreme temperature swings and strong steppe winds."
    },
    "Almaty 🏔️": {
        "bg_url": "https://images.unsplash.com/photo-1589561287413-568fb8d022fa?q=80&w=1200",
        "shift": 8.5,  
        "desc": "Southern metropolis nestled near the Tien Shan mountains. Much milder and warmer climate."
    },
    "Shymkent ☀️": {
        "bg_url": "https://images.unsplash.com/photo-1628131341065-983b632fa1bf?q=80&w=1200",
        "shift": 12.0, 
        "desc": "The sunniest and warmest major city in Kazakhstan. Hot summers and short winters."
    },
    "Petropavl ❄️": {
        "bg_url": "https://images.unsplash.com/photo-1608958416712-4fbff73d4060?q=80&w=1200",
        "shift": -3.5, 
        "desc": "Northern region. Heavily influenced by Siberian air masses. Crisp, cold climate."
    },
    "Aktau 🌊": {
        "bg_url": "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?q=80&w=1200",
        "shift": 5.0,  
        "desc": "Caspian Sea coast. Maritime influence makes winters softer and summers breezy."
    }
}

# --- ВЫБОР ГОРОДА НА БОКОВОЙ ПАНЕЛИ ---
st.sidebar.header("📍 Location & Controls")
selected_city = st.sidebar.selectbox("Choose a City:", list(CITIES.keys()))

# Получаем данные выбранного города
city_data = CITIES[selected_city]

# --- ДИНАМИЧЕСКИЙ ЗАДНИЙ ФОН (CSS Инъекция) ---
# Этот блок магии меняет картинку на заднем фоне сайта при переключении города
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.85)), 
                    url("{city_data['bg_url']}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1, h2, h3, p, span, label {{
        color: #ffffff !important;
    }}
    .stSlider label {{
        color: #ffffff !important;
    }}
    .css-1dp5639, .stMetric {{
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🌤️ Kazakhstan Weather AI Simulator")
st.caption(f"Currently simulating: **{selected_city}**")
st.write(f"*{city_data['desc']}*")
st.markdown("---")

# Слайдеры на боковой панели
st.sidebar.markdown("---")
st.sidebar.markdown("**Tune Atmospheric Parameters:**")
humidity = st.sidebar.slider('Relative Humidity (%)', 10, 100, 55)
pressure = st.sidebar.slider('Surface Pressure (kPa)', 90, 110, 101)
wind_speed = st.sidebar.slider('Wind Speed (m/s)', 0.0, 25.0, 6.5)

# Feature Engineering
weather_index = humidity * wind_speed

# --- МОНИТОР ДАННЫХ ---
st.subheader("📊 Live Parameter Monitor")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Humidity", f"{humidity} %")
with col2:
    st.metric("Pressure", f"{pressure} kPa")
with col3:
    st.metric("Wind Speed", f"{wind_speed} m/s")

# Подготовка данных для модели
input_data = pd.DataFrame([[humidity, pressure, wind_speed, weather_index]], 
                          columns=['humidity', 'pressure', 'wind_speed', 'weather_index'])

st.markdown("---")

# --- ПРОГНОЗ МОДЕЛИ ---
try:
    model = joblib.load('weather_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    if st.sidebar.button('🚀 Run AI Prediction'):
        with st.spinner('AI is processing regional climate vectors...'):
            time.sleep(0.6)
            
            # Базовый прогноз модели (заданный Астаной)
            input_scaled = scaler.transform(input_data)
            base_prediction = model.predict(input_scaled)[0]
            
            # Применяем климатический сдвиг для выбранного города
            final_prediction = base_prediction + city_data['shift']
        
        st.subheader(f"🎯 AI Forecast for {selected_city}")
        st.success(f"### Predicted Temperature: {final_prediction:.2f} °C")
        
        # Визуальные эффекты
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
with st.expander("ℹ️ How does the Multi-City feature work?"):
    st.write("""
    This simulator uses a base Core Machine Learning Pipeline trained on long-term NASA telemetry. 
    To scale it nationwide, we integrate dynamic climate vectors (regional offsets) calibrated against 
    the geographic coordinates and thermal baselines of each specific Kazakhstani region.
    """)
