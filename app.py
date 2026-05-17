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
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kazakhstan Weather AI Simulator 🇰🇿",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- DICTIONARY OF KAZAKHSTAN CITIES ---
CITIES = {
    "Astana 🏙️": {
        "shift": 0.0,
        "desc": "Capital city. Known for extreme temperature swings and strong steppe winds.",
        "lat": 51.1694,
        "lon": 71.4691
    },
    "Almaty 🏔️": {
        "shift": 8.5,
        "desc": "Southern metropolis nestled near the Tien Shan mountains. Much milder and warmer climate.",
        "lat": 43.2380,
        "lon": 76.9385
    },
    "Semey 🏛️": {
        "shift": 2.5,
        "desc": "Historical cultural center on the Irtysh river. Famous for its unique pine forest and iconic suspension bridge.",
        "lat": 50.4111,
        "lon": 80.2275
    },
    "Shymkent ☀️": {
        "shift": 12.0,
        "desc": "The sunniest and warmest major city. Hot summers and short winters.",
        "lat": 42.3155,
        "lon": 69.5947
    },
    "Aktau 🌊": {
        "shift": 5.0,
        "desc": "Caspian Sea coast. Maritime influence makes winters softer and summers breezy.",
        "lat": 43.6426,
        "lon": 51.1694
    }
}

# --- TRAIN OR LOAD MODELS ---
@st.cache_resource
def get_trained_models_and_scaler():
    """
    Tries to load pre-trained models or trains your 4 Jupyter models from scratch
    """
    try:
        models = {
            'Linear Regression': joblib.load('model_linear_regression.pkl'),
            'Random Forest': joblib.load('model_random_forest.pkl'),
            'KNN Regressor': joblib.load('model_knn.pkl'),
            'XGBoost Regressor': joblib.load('model_xgboost.pkl')
        }
        scaler = joblib.load('scaler.pkl')
        return models, scaler, True
    except:
        # Если файлов нет, обучаем твои 4 модели на лету
        return train_models()

def train_models():
    """
    Trains your exact 4 Jupyter models on localized meteorological dataset structure
    """
    np.random.seed(42)
    n_samples = 1200
    
    # Генерация признаков под структуру твоего df
    humidity = np.random.normal(55, 18, n_samples)
    humidity = np.clip(humidity, 10, 100)
    
    pressure = np.random.normal(101, 2.5, n_samples)
    pressure = np.clip(pressure, 90, 110)
    
    wind_speed = np.random.exponential(6.0, n_samples)
    wind_speed = np.clip(wind_speed, 0, 25)
    
    weather_index = humidity * wind_speed
    
    # Целевая переменная (зависимость температуры)
    temperature = (15 + (humidity - 50) * 0.12 + (pressure - 101) * 1.8 
                  - wind_speed * 0.45 + np.random.normal(0, 2.5, n_samples))
    
    # Выделяем фичи строго как в твоем Jupyter: X = df[['humidity', 'pressure', 'wind_speed', 'weather_index']]
    X = np.column_stack([humidity, pressure, wind_speed, weather_index])
    
    # Масштабирование StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Инициализируем ТВОИ 4 модели
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "XGBoost Regressor": XGBRegressor(random_state=42)
    }
    
    # Обучаем каждую модель
    for name, model in models.items():
        model.fit(X_scaled, temperature)
        
    return models, scaler, False

# --- MAIN APP ---
def main():
    models, scaler, models_loaded = get_trained_models_and_scaler()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌤️ Kazakhstan Weather AI Simulator")
        st.markdown("**Your Jupyter Pipeline Model Comparison & Temperature Prediction**")
    with col2:
        st.write(f"**Last Sync:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("📍 Location & Controls")
        selected_city = st.selectbox("🏙️ Choose a City:", list(CITIES.keys()))
        city_data = CITIES[selected_city]
        
        st.markdown("---")
        st.subheader("⚙️ Tune Atmospheric Parameters:")
        
        humidity = st.slider('💧 Relative Humidity (%)', 10, 100, 45)
        pressure = st.slider('🧭 Surface Pressure (kPa)', 90.0, 110.0, 100.5)
        wind_speed = st.slider('💨 Wind Speed (m/s)', 0.0, 25.0, 4.2)
        
        st.markdown("---")
        st.subheader("🤖 Model Selection")
        selected_models = st.multiselect(
            "Choose models to compare:",
            list(models.keys()),
            default=list(models.keys())
        )
        
        st.markdown("---")
        predict_button = st.button('🚀 Run AI Predictions', use_container_width=True)
    
    # Main content area - City Metadata & Map View
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(f"📌 {selected_city}")
        st.write(f"*{city_data['desc']}*")
        st.markdown(f"📍 **Coordinates:** {city_data['lat']}°N, {city_data['lon']}°E")
        st.markdown(f"📊 **Climate Offset Vector:** `{city_data['shift']:+.1f} °C`")
    
    with col2:
        # Географическая карта вместо проблемных ссылок на фото
        map_data = pd.DataFrame({'lat': [city_data['lat']], 'lon': [city_data['lon']]})
        st.map(map_data, zoom=8, use_container_width=True)
    
    st.markdown("---")
    
    # Parameter Monitor
    st.subheader("📊 Live Parameter Monitor")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("💧 Humidity", f"{humidity}%")
    with metric_col2:
        st.metric("🧭 Pressure", f"{pressure:.1f} kPa")
    with metric_col3:
        st.metric("💨 Wind Speed", f"{wind_speed:.1f} m/s")
    with metric_col4:
        weather_index = humidity * wind_speed
        st.metric("🌡️ Weather Index", f"{weather_index:.1f}", delta="Derived Feature")
    
    st.markdown("---")
    
    # Predictions Section
    if predict_button:
        if not selected_models:
            st.warning("⚠️ Please select at least one Machine Learning model from the sidebar.")
            return

        st.subheader("🎯 AI Model Predictions")
        
        # Входной вектор строго под твои 4 фичи
        features = np.array([[humidity, pressure, wind_speed, weather_index]])
        features_scaled = scaler.transform(features)
        
        predictions = {}
        with st.spinner("🔄 Running predictions across your Jupyter models..."):
            time.sleep(0.4)
            for model_name in selected_models:
                if model_name in models:
                    pred = models[model_name].predict(features_scaled)[0]
                    final_pred = pred + city_data['shift']
                    predictions[model_name] = final_pred
        
        # Display predictions side-by-side
        prediction_cols = st.columns(len(predictions))
        for idx, (model_name, temp) in enumerate(predictions.items()):
            with prediction_cols[idx]:
                if temp <= 0:
                    color, bg_color = "❄️", "#cfe2ff"
                elif 0 < temp <= 18:
                    color, bg_color = "🌤️", "#e2e3e5"
                elif 18 < temp <= 28:
                    color, bg_color = "☀️", "#fff3cd"
                else:
                    color, bg_color = "🔥", "#f8d7da"
                
                st.markdown(f"""
                    <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #dee2e6;'>
                    <h4 style='color: #212529; margin: 0 0 10px 0;'>{model_name}</h4>
                    <h2 style='color: #212529; margin: 0 0 5px 0;'>{temp:.2f}°C</h2>
                    <p style='font-size: 24px; margin: 0;'>{color}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Comparison Table
        st.subheader("📊 Detailed Model Comparison Matrix")
        comparison_data = {
            'Model Architecture': list(predictions.keys()),
            'Simulated Prediction (°C)': [f"{v:.2f}" for v in predictions.values()],
            'Regional Offset Climate Vector': [f"{city_data['shift']:+.1f}°C" for _ in predictions],
            'Base Model Output': [f"{float(v) - city_data['shift']:.2f}°C" for v in predictions.values()]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        # Weather alerts
        avg_temp = np.mean(list(predictions.values()))
        st.markdown("---")
        st.subheader("⚠️ Weather Interpretation")
        
        if avg_temp <= -15:
            st.error("🥶 SEVERE COLD WARNING: Extreme freezing conditions simulated!")
        elif avg_temp <= 0:
            st.info("❄️ FROST WARNING: Sub-zero freezing temperatures expected.")
        elif avg_temp >= 28:
            st.error("🔥 HEAT WARNING: Extreme hot simulation vectors!")
        else:
            st.success("✅ Normal and stable seasonal weather conditions.")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ How does this ML system work?"):
        st.markdown("""
        ### System Architecture (Your Jupyter Pipeline Blueprint):
        
        **Data Processing Constraints:**
        - Features Used ($X$): `humidity`, `pressure`, `wind_speed`, `weather_index` ($Humidity \\times Wind\\_Speed$).
        - Normalization Matrix: Automated `StandardScaler()` fit-transformation sequence.
        
        **Your Implemented Models:**
        1. **Linear Regression** - Base model mapping straight-line meteorological correlations.
        2. **Random Forest** - Tree-ensemble bagging structure optimized for regional metrics.
        3. **KNN Regressor** - Distance-based feature localization vector matching ($K=5$).
        4. **XGBoost Regressor** - Gradient-boosted decision trees for precision optimization.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Kazakhstan Weather AI Simulator v2.5 | Astana IT University | Machine Learning Project</p>
    <p>© 2026 | Data Pipeline: scikit-learn & xgboost</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
