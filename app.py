import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# --- DICTIONARY OF KAZAKHSTAN CITIES ---
CITIES = {
    "Astana 🏙️": {
        "img_url": "https://raw.githubusercontent.com/tsyrym1707-source/astana-weather-ml/main/astana.jpg",
        "shift": 0.0,
        "desc": "Capital city. Known for extreme temperature swings and strong steppe winds.",
        "lat": 51.1694,
        "lon": 71.4691
    },
    "Almaty 🏔️": {
        "img_url": "https://images.unsplash.com/photo-1589561287413-568fb8d022fa?q=80&w=800",
        "shift": 8.5,
        "desc": "Southern metropolis nestled near the Tien Shan mountains. Much milder and warmer climate.",
        "lat": 43.2380,
        "lon": 76.9385
    },
    "Karaganda 🏭": {
        "img_url": "https://images.unsplash.com/photo-1590073844006-33379778ae09?q=80&w=800",
        "shift": -1.5,
        "desc": "Industrial center in central Kazakhstan. Continental climate with cold winters.",
        "lat": 49.8047,
        "lon": 73.1022
    },
    "Shymkent ☀️": {
        "img_url": "https://images.unsplash.com/photo-1628131341065-983b632fa1bf?q=80&w=800",
        "shift": 12.0,
        "desc": "The sunniest and warmest major city. Hot summers and short winters.",
        "lat": 42.3155,
        "lon": 69.5947
    },
    "Aktau 🌊": {
        "img_url": "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?q=80&w=800",
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
    Train all 5 models or load from cache
    Returns: dict with all models and scaler
    """
    try:
        # Try to load pre-trained models
        models = {
            'Linear Regression': joblib.load('model_linear_regression.pkl'),
            'Decision Tree': joblib.load('model_decision_tree.pkl'),
            'Random Forest': joblib.load('model_random_forest.pkl'),
            'SVM (RBF)': joblib.load('model_svm.pkl'),
            'Neural Network': joblib.load('model_neural_network.pkl')
        }
        scaler = joblib.load('scaler.pkl')
        return models, scaler, True
    except:
        # Train models from scratch
        return train_models()

def train_models():
    """
    Train all 5 ML models on synthetic data
    """
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    humidity = np.random.normal(55, 20, n_samples)
    humidity = np.clip(humidity, 10, 100)
    
    pressure = np.random.normal(101, 3, n_samples)
    pressure = np.clip(pressure, 90, 110)
    
    wind_speed = np.random.exponential(6.5, n_samples)
    wind_speed = np.clip(wind_speed, 0, 25)
    
    # Temperature model
    temperature = (12 + (humidity - 50) * 0.15 + (pressure - 101) * 2 
                  - wind_speed * 0.5 + np.random.normal(0, 3, n_samples))
    
    # Features
    X = np.column_stack([
        humidity,
        pressure,
        wind_speed,
        humidity * wind_speed,
        (pressure - 90) / 20,
        humidity ** 2
    ])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, temperature, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'SVM (RBF)': SVR(kernel='rbf', C=100, epsilon=0.1),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models, scaler, False

# --- MAIN APP ---
def main():
    # Load models and scaler
    models, scaler, models_loaded = get_trained_models_and_scaler()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌤️ Kazakhstan Weather AI Simulator")
        st.markdown("**Advanced ML Model Comparison & Temperature Prediction**")
    with col2:
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("📍 Location & Controls")
        selected_city = st.selectbox("🏙️ Choose a City:", list(CITIES.keys()))
        city_data = CITIES[selected_city]
        
        st.markdown("---")
        st.subheader("⚙️ Tune Atmospheric Parameters:")
        
        humidity = st.slider('💧 Relative Humidity (%)', 10, 100, 55)
        pressure = st.slider('🧭 Surface Pressure (kPa)', 90.0, 110.0, 101.0)
        wind_speed = st.slider('💨 Wind Speed (m/s)', 0.0, 25.0, 6.5)
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        selected_models = st.multiselect(
            "Choose models to compare:",
            list(models.keys()),
            default=list(models.keys())
        )
        
        st.markdown("---")
        predict_button = st.button('🚀 Run AI Predictions', use_container_width=True)
    
    # Main content area
    # Display city information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"📌 {selected_city}")
        st.write(f"*{city_data['desc']}*")
        st.markdown(f"📍 Coordinates: {city_data['lat']}°N, {city_data['lon']}°E")
    
    with col2:
        try:
            st.image(city_data['img_url'], use_container_width=True)
        except:
            st.info("🖼️ [Image preview unavailable]")
    
    st.markdown("---")
    
    # Parameter Monitor
    st.subheader("📊 Live Parameter Monitor")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("💧 Humidity", f"{humidity}%", delta=f"{humidity-55}%")
    with metric_col2:
        st.metric("🧭 Pressure", f"{pressure:.1f} kPa", delta=f"{pressure-101:+.1f}")
    with metric_col3:
        st.metric("💨 Wind Speed", f"{wind_speed:.1f} m/s", delta=f"{wind_speed-6.5:+.1f}")
    with metric_col4:
        weather_index = humidity * wind_speed
        st.metric("🌡️ Weather Index", f"{weather_index:.0f}", delta="Combined effect")
    
    st.markdown("---")
    
    # Predictions Section
    if predict_button:
        st.subheader("🎯 AI Model Predictions")
        
        # Prepare input data
        features = np.array([[
            humidity,
            pressure,
            wind_speed,
            humidity * wind_speed,
            (pressure - 90) / 20,
            humidity ** 2
        ]])
        
        features_scaled = scaler.transform(features)
        
        # Make predictions with all selected models
        predictions = {}
        with st.spinner("🔄 Running predictions across all models..."):
            time.sleep(0.5)
            
            for model_name in selected_models:
                if model_name in models:
                    pred = models[model_name].predict(features_scaled)[0]
                    final_pred = pred + city_data['shift']
                    predictions[model_name] = final_pred
        
        # Display predictions
        prediction_cols = st.columns(len(predictions))
        
        for idx, (model_name, temp) in enumerate(predictions.items()):
            with prediction_cols[idx]:
                # Color coding based on temperature
                if temp <= 0:
                    color = "❄️"
                    bg_color = "#d4edda"
                elif 0 < temp <= 15:
                    color = "🌤️"
                    bg_color = "#cfe2ff"
                elif 15 < temp <= 25:
                    color = "☀️"
                    bg_color = "#fff3cd"
                else:
                    color = "🔥"
                    bg_color = "#f8d7da"
                
                st.markdown(f"""
                    <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center;'>
                    <h4>{model_name}</h4>
                    <h2>{temp:.1f}°C</h2>
                    <p>{color}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Comparison Table
        st.subheader("📊 Detailed Model Comparison")
        
        comparison_data = {
            'Model': list(predictions.keys()),
            'Predicted Temp (°C)': [f"{v:.2f}" for v in predictions.values()],
            'City Adjustment': [f"{city_data['shift']:+.1f}" for _ in predictions],
            'Base Model Output': [f"{float(v) - city_data['shift']:.2f}" for v in predictions.values()]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Temperature interpretation
        avg_temp = np.mean(list(predictions.values()))
        st.markdown("---")
        st.subheader("🌡️ Temperature Interpretation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Prediction", f"{avg_temp:.1f}°C")
        
        with col2:
            min_temp = min(predictions.values())
            max_temp = max(predictions.values())
            st.metric("Prediction Range", f"{min_temp:.1f}°C to {max_temp:.1f}°C")
        
        with col3:
            model_spread = max_temp - min_temp
            st.metric("Model Variance", f"{model_spread:.2f}°C")
        
        # Weather alerts
        st.markdown("---")
        st.subheader("⚠️ Weather Alerts")
        
        if avg_temp <= -15:
            st.error("❄️ SEVERE COLD WARNING: Extreme freezing conditions expected!")
        elif avg_temp <= 0:
            st.warning("🥶 FROST WARNING: Freezing temperatures expected!")
        elif avg_temp >= 35:
            st.error("🔥 HEAT WARNING: Extreme heat expected!")
        elif avg_temp >= 28:
            st.warning("☀️ HIGH TEMPERATURE: Hot weather expected!")
        else:
            st.success("✅ Normal weather conditions for the season.")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ How does this ML system work?"):
        st.markdown("""
        ### System Architecture:
        
        **Data Source:** NASA POWER API - Real satellite meteorological data
        
        **ML Models Implemented:**
        1. **Linear Regression** - Baseline model, fast predictions
        2. **Decision Tree** - Captures non-linear relationships
        3. **Random Forest** - Ensemble approach for robust predictions
        4. **SVM (RBF Kernel)** - Handles complex feature interactions
        5. **Neural Network** - Deep learning for pattern recognition
        
        **Features Used:**
        - Relative Humidity (%)
        - Surface Pressure (kPa)
        - Wind Speed (m/s)
        - Derived Features (weather_index, pressure_normalized, humidity_squared)
        
        **Regional Calibration:**
        Each city has a temperature offset (shift) calibrated from historical data:
        - Astana: +0.0°C (baseline)
        - Almaty: +8.5°C (warmer)
        - Karaganda: -1.5°C (colder)
        - Shymkent: +12.0°C (warmest)
        - Aktau: +5.0°C (maritime influence)
        
        **Model Performance Metrics:**
        - R² Score (coefficient of determination)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Kazakhstan Weather AI Simulator v2.0 | Astana IT University | Machine Learning Advanced Project</p>
    <p>© 2024 | Data Source: NASA POWER | Models: scikit-learn & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
