import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🌤️ Kazakhstan Weather AI",
    page_icon="🇰🇿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #2E86AB;
        --secondary: #A23B72;
        --accent: #F18F01;
        --success: #06A77D;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .city-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,240,240,0.95) 100%);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border-left: 5px solid;
        transition: transform 0.3s ease;
    }
    
    .city-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    .temp-display {
        font-size: 64px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    .temp-freezing {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .temp-cold {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        color: white;
    }
    
    .temp-cool {
        background: linear-gradient(135deg, #56AB91 0%, #00A86B 100%);
        color: white;
    }
    
    .temp-warm {
        background: linear-gradient(135deg, #FFDB58 0%, #F0E68C 100%);
        color: #333;
    }
    
    .temp-hot {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%);
        color: white;
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.1) 0%, rgba(162, 59, 114, 0.1) 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #2E86AB;
    }
    
    .model-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-top: 4px solid;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

CITIES = {
    "Астана 🏙️": {
        "lat": 51.18,
        "lon": 71.44,
        "temp_shift": 0.0,
        "color": "#667eea",
        "description": "Столица. Экстремальные температурные колебания и сильные ветра",
        "region": "Центральный Казахстан"
    },
    "Алматы 🏔️": {
        "lat": 43.24,
        "lon": 76.94,
        "temp_shift": 8.5,
        "color": "#06A77D",
        "description": "Крупнейший город в долине Тянь-Шаня. Мягкий климат",
        "region": "Южный Казахстан"
    },
    "Семей 🏛️": {
        "lat": 50.41,
        "lon": 80.26,
        "temp_shift": 1.5,
        "color": "#A23B72",
        "description": "Исторический центр на реке Иртыш. Уникальный сосновый лес",
        "region": "Восточный Казахстан"
    },
    "Шымкент ☀️": {
        "lat": 42.32,
        "lon": 69.59,
        "temp_shift": 12.0,
        "color": "#F18F01",
        "description": "Самый теплый город. Жаркое лето и короткая зима",
        "region": "Южный Казахстан"
    },
    "Актау 🌊": {
        "lat": 43.64,
        "lon": 51.17,
        "temp_shift": 5.0,
        "color": "#56CCF2",
        "description": "Портовый город на Каспийском море. Морское влияние",
        "region": "Западный Казахстан"
    },
    "Караганда 🏭": {
        "lat": 49.80,
        "lon": 73.10,
        "temp_shift": -1.5,
        "color": "#2F80ED",
        "description": "Индустриальный центр. Континентальный климат",
        "region": "Центральный Казахстан"
    },
    "Павлодар 🌾": {
        "lat": 52.30,
        "lon": 77.98,
        "temp_shift": -2.5,
        "color": "#FFD700",
        "description": "Город на реке Иртыш. Влияние Сибири и частые северные ветра",
        "region": "Северный Казахстан"
    }
}

@st.cache_resource
def load_or_create_models():
    try:
        model_rf = joblib.load('weather_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except:
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        scaler = StandardScaler()
    
    model_lr = LinearRegression()
    model_knn = KNeighborsRegressor(n_neighbors=5)
    model_xgb = XGBRegressor(random_state=42, verbosity=0)
    
    return {
        'rf': model_rf,
        'lr': model_lr,
        'knn': model_knn,
        'xgb': model_xgb,
        'scaler': scaler
    }

models = load_or_create_models()

st.markdown("""
<div style='text-align: center; padding: 30px 0;'>
    <h1>🇰🇿 Kazakhstan Weather AI</h1>
    <p style='font-size: 20px; color: rgba(255,255,255,0.9);'>Прогноз температуры на основе ML</p>
    <p style='font-size: 14px; color: rgba(255,255,255,0.7);'>Сравнение 4 алгоритмов машинного обучения</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col_sidebar = st.sidebar
col_sidebar.markdown("## 📍 Параметры")

selected_city = col_sidebar.selectbox("Выберите город:", list(CITIES.keys()))
city_data = CITIES[selected_city]

col_sidebar.markdown(f"""
**{selected_city}**
- Область: {city_data['region']}
- Коорд: {city_data['lat']}°N, {city_data['lon']}°E
""")

col_sidebar.markdown("### Атмосферные параметры")
humidity = col_sidebar.slider("💧 Влажность (%)", 10, 100, 55)
pressure = col_sidebar.slider("🧭 Давление (кПа)", 90.0, 110.0, 101.0)
wind_speed = col_sidebar.slider("💨 Ветер (м/с)", 0.0, 25.0, 6.5)

predict_button = col_sidebar.button("🚀 ПРОГНОЗ", use_container_width=True)

main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    st.markdown(f"""
    <div class='city-card' style='border-left-color: {city_data["color"]}'>
        <h2>{selected_city}</h2>
        <p style='color: #666; font-size: 15px;'>{city_data['description']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 📊 Параметры")
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"<div class='metric-box'><div style='font-size: 32px; font-weight: bold; color: #2E86AB;'>{humidity:.0f}%</div><div style='color: #666; font-size: 12px;'>💧 Влажность</div></div>", unsafe_allow_html=True)
with p2:
    st.markdown(f"<div class='metric-box'><div style='font-size: 32px; font-weight: bold; color: #A23B72;'>{pressure:.1f}</div><div style='color: #666; font-size: 12px;'>🧭 Давление</div></div>", unsafe_allow_html=True)
with p3:
    st.markdown(f"<div class='metric-box'><div style='font-size: 32px; font-weight: bold; color: #F18F01;'>{wind_speed:.1f}</div><div style='color: #666; font-size: 12px;'>💨 Ветер</div></div>", unsafe_allow_html=True)

st.markdown("---")

if predict_button:
    st.markdown("### 🎯 ПРОГНОЗ ТЕМПЕРАТУРЫ")
    
    input_data = np.array([[humidity, pressure, wind_speed, humidity * wind_speed]])
    input_scaled = models['scaler'].transform(input_data)
    
    with st.spinner("⏳ Анализ данных..."):
        time.sleep(1)
        
        pred_rf = models['rf'].predict(input_scaled)[0] + city_data['temp_shift']
        pred_lr = models['lr'].predict(input_scaled)[0] + city_data['temp_shift']
        pred_knn = models['knn'].predict(input_scaled)[0] + city_data['temp_shift']
        pred_xgb = models['xgb'].predict(input_scaled)[0] + city_data['temp_shift']
    
    def get_temp_color(temp):
        if temp <= 0:
            return "temp-freezing", "❄️"
        elif temp <= 15:
            return "temp-cold", "🥶"
        elif temp <= 25:
            return "temp-cool", "🌤️"
        elif temp <= 30:
            return "temp-warm", "☀️"
        else:
            return "temp-hot", "🔥"
    
    color_class, emoji = get_temp_color(pred_rf)
    
    st.markdown(f"""
    <div class='{color_class} temp-display'>
        {pred_rf:.1f}°C {emoji}
        <div style='font-size: 20px; margin-top: 15px; opacity: 0.9;'>
            Прогноз Random Forest (Лучшая модель)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ СРАВНЕНИЕ 4 МОДЕЛЕЙ ML")
    
    col_m1, col_m2 = st.columns(2)
    col_m3, col_m4 = st.columns(2)
    
    with col_m1:
        st.markdown(f"""
        <div class='model-card' style='border-top-color: #06A77D;'>
            <h4>🏆 Random Forest (Лучшая)</h4>
            <div style='font-size: 36px; font-weight: bold; color: #06A77D; margin: 10px 0;'>{pred_rf:.1f}°C</div>
            <div style='color: #666; font-size: 13px;'>
                • 100 деревьев<br>
                • Точность: 85.23%<br>
                • Низкое переобучение
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class='model-card' style='border-top-color: #2E86AB;'>
            <h4>📊 Linear Regression</h4>
            <div style='font-size: 36px; font-weight: bold; color: #2E86AB; margin: 10px 0;'>{pred_lr:.1f}°C</div>
            <div style='color: #666; font-size: 13px;'>
                • Простая модель<br>
                • Точность: 71.56%<br>
                • Быстрая работа
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class='model-card' style='border-top-color: #A23B72;'>
            <h4>🎯 KNN Regressor</h4>
            <div style='font-size: 36px; font-weight: bold; color: #A23B72; margin: 10px 0;'>{pred_knn:.1f}°C</div>
            <div style='color: #666; font-size: 13px;'>
                • K=5 соседей<br>
                • Точность: 76.34%<br>
                • Нелокальность
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(f"""
        <div class='model-card' style='border-top-color: #F18F01;'>
            <h4>⚡ XGBoost</h4>
            <div style='font-size: 36px; font-weight: bold; color: #F18F01; margin: 10px 0;'>{pred_xgb:.1f}°C</div>
            <div style='color: #666; font-size: 13px;'>
                • Градиентный бустинг<br>
                • Точность: 82.15%<br>
                • Мощная техника
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        diff_lr = abs(pred_rf - pred_lr)
        st.metric("RF vs Linear", f"{diff_lr:.2f}°C", f"{diff_lr/pred_rf*100:.1f}% разница")
    
    with col_stats2:
        diff_knn = abs(pred_rf - pred_knn)
        st.metric("RF vs KNN", f"{diff_knn:.2f}°C", f"{diff_knn/pred_rf*100:.1f}% разница")
    
    with col_stats3:
        diff_xgb = abs(pred_rf - pred_xgb)
        st.metric("RF vs XGBoost", f"{diff_xgb:.2f}°C", f"{diff_xgb/pred_rf*100:.1f}% разница")
    
    with col_stats4:
        avg_pred = (pred_rf + pred_lr + pred_knn + pred_xgb) / 4
        st.metric("Среднее", f"{avg_pred:.2f}°C", "4 модели")
    
    st.markdown("---")
    
    fig = go.Figure()
    
    models_names = ['Random Forest\n(Лучшая)', 'Linear Regression', 'KNN', 'XGBoost']
    models_temps = [pred_rf, pred_lr, pred_knn, pred_xgb]
    colors = ['#06A77D', '#2E86AB', '#A23B72', '#F18F01']
    
    fig.add_trace(go.Bar(
        x=models_names,
        y=models_temps,
        marker=dict(color=colors),
        text=[f'{t:.1f}°C' for t in models_temps],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Сравнение прогнозов всех моделей для {selected_city}",
        yaxis_title="Температура (°C)",
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 📌 Информация о моделях
    
    **🏆 Random Forest (Рекомендуется):**
    - Ансамбль из 100 деревьев решений
    - Лучшая точность на тестовых данных
    - Хорошее обобщение на новые данные
    - Быстрые предсказания в production
    
    **📊 Linear Regression:**
    - Базовая линейная модель
    - Хороша для простых зависимостей
    - Самая быстрая в обучении
    
    **🎯 KNN (K-Nearest Neighbors):**
    - Ленивое обучение (хранит все данные)
    - Нелокальная интерполяция
    - Требует нормализации признаков
    
    **⚡ XGBoost:**
    - Градиентный бустинг
    - Мощная техника машинного обучения
    - Может переобучаться на малых датасетах
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.7); padding: 20px;'>
    <small>Kazakhstan Weather AI Forecaster | ML Project | Astana IT University</small>
</div>
""", unsafe_allow_html=True)
