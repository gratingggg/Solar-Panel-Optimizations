import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import joblib

sys.path.append('src')

from weather_api import WeatherAPIClient
from forecasting_engine import ForecastingEngine
from tilt_angle_estimator import TiltAngleEstimator
from config import UI_CONFIG, MODEL_DIR
from streamlit_js_eval import get_geolocation

st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #1f2937;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #1f2937;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False

@st.cache_resource
def load_resources():
    try:
        engine = ForecastingEngine()
        weather_client = WeatherAPIClient()
        tilt_estimator = TiltAngleEstimator()
        return engine, weather_client, tilt_estimator, None
    except Exception as e:
        return None, None, None, str(e)

engine, weather_client, tilt_estimator, load_error = load_resources()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Solar Forecasting", "Model Comparison"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption("Solar Energy Forecasting System")
st.sidebar.caption("Powered by Machine Learning")

if page == "Solar Forecasting":
    st.markdown('<div class="main-header">Solar Energy Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered 48-Hour Power Generation Predictions</div>', unsafe_allow_html=True)
    
    if load_error or engine is None or engine.model is None:
        st.error(f"""
        **Model Not Found**  
        Please train the models first by running: `notebooks/solar_forecasting_models.ipynb`  
        {f'Error: {load_error}' if load_error else ''}
        """)
        st.stop()
    
  
    st.header("System Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Panel Specifications")
        panel_width = st.number_input("Width (m)", 0.5, 5.0, 1.6, 0.1)
        panel_height = st.number_input("Height (m)", 0.5, 5.0, 1.0, 0.1)
        rated_capacity = st.number_input("Capacity (W)", 50, 1000, 300, 10)
    
    with col2:
        st.subheader("Geographic Location")
        loc_method = st.radio("Location Source", ["Manual Input", "Device GPS"], horizontal=True)
        
        default_lat, default_lon = 23.0225, 72.5714
        
        if loc_method == "Device GPS":
            loc = get_geolocation()
            if loc:
                if 'coords' in loc:
                    default_lat = loc['coords']['latitude']
                    default_lon = loc['coords']['longitude']
                    st.success(f"GPS Lock: {default_lat:.4f}, {default_lon:.4f}")
                elif 'error' in loc:
                    st.error(f"GPS Error: {loc['error'].get('message', 'Unknown error')}")
                    st.info("Falling back to manual input.")
                else:
                    st.info("Waiting for GPS signal... Please allow location access in your browser.")
            else:
                st.info("Searching for GPS... Ensure location is enabled on your device.")
        
        latitude = st.number_input("Latitude", -90.0, 90.0, float(default_lat), 0.0001, format="%.4f")
        longitude = st.number_input("Longitude", -180.0, 180.0, float(default_lon), 0.0001, format="%.4f")
    
    with col3:
        st.subheader("Panel Tilt Angle")
        tilt_method = st.radio("Input Method", ["Manual", "Image Upload"], horizontal=True)
        
        if tilt_method == "Manual":
            tilt_angle = st.slider("Tilt Angle (°)", 0, 90, int(abs(latitude)))
            tilt_confidence = 1.0
        else:
            uploaded_file = st.file_uploader("Upload Panel Image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                temp_path = Path("cache") / uploaded_file.name
                temp_path.parent.mkdir(exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                tilt_angle, tilt_confidence = tilt_estimator.estimate_tilt_from_image(str(temp_path))
                st.success(f"Estimated: {tilt_angle:.1f}° (Confidence: {tilt_confidence:.2f})")
                st.image(uploaded_file, width=200)
            else:
                tilt_angle = int(abs(latitude))
                tilt_confidence = 0.5
                st.info(f"Default: {tilt_angle}°")
    
    st.divider()
    
    if st.button("Generate 48-Hour Forecast", type="primary", use_container_width=True):
        with st.spinner("Fetching weather data and generating forecast..."):
            try:
                weather_forecast = weather_client.get_forecast(latitude, longitude, hours=48)
                current_weather = weather_client.get_current_weather(latitude, longitude)
                
                if weather_forecast is None:
                    st.error("Failed to fetch weather data. Check API key and internet connection.")
                    st.stop()
                
                panel_specs = {
                    'width': panel_width,
                    'height': panel_height,
                    'rated_capacity': rated_capacity,
                    'tilt_angle': tilt_angle,
                    'latitude': latitude,
                    'longitude': longitude
                }
                
                forecast_df = engine.generate_forecast(weather_forecast, panel_specs, hours=48)
                
                current_month = datetime.now().month
                season = (current_month % 12 + 3) // 3
                
                tilt_recommendation = engine.recommend_tilt_adjustment(tilt_angle, latitude, season)
                maintenance_status = engine.detect_maintenance_needs(forecast_df, panel_specs)
                weather_impact = engine.analyze_weather_impact(forecast_df)
                
                st.session_state.forecast_generated = True
                st.session_state.forecast_df = forecast_df
                st.session_state.current_weather = current_weather
                st.session_state.tilt_recommendation = tilt_recommendation
                st.session_state.maintenance_status = maintenance_status
                st.session_state.weather_impact = weather_impact
                st.session_state.panel_specs = panel_specs
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.stop()

    if st.session_state.forecast_generated:
        forecast_df = st.session_state.forecast_df
        current_weather = st.session_state.current_weather
        tilt_recommendation = st.session_state.tilt_recommendation
        maintenance_status = st.session_state.maintenance_status
        weather_impact = st.session_state.weather_impact
        
        st.divider()

        st.header("Current Weather Conditions")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{current_weather['temperature']:.1f}°C")
        col2.metric("Humidity", f"{current_weather['humidity']:.0f}%")
        col3.metric("Cloud Cover", f"{current_weather['cloud_cover']:.0f}%")
        col4.metric("Wind Speed", f"{current_weather['wind_speed']:.1f} m/s")
        
        if current_weather.get('is_fallback'):
            st.warning("Using fallback weather data (API unavailable)")
        
        st.divider()
        
        st.header("48-Hour Power Generation Forecast")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['upper_bound'],
            mode='lines', name='Upper Bound', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['lower_bound'],
            mode='lines', name='Confidence Interval', line=dict(width=0),
            fillcolor='rgba(31, 119, 180, 0.2)', fill='tonexty'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['predicted_power'],
            mode='lines', name='Predicted Power', line=dict(color='#1f77b4', width=3)
        ))
        fig.update_layout(
            xaxis_title="Time", yaxis_title="Power (W)",
            hovermode='x unified', height=500, template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Energy", f"{weather_impact['total_expected_energy_kwh']:.2f} kWh")
        col2.metric("Peak Power", f"{weather_impact['peak_power']:.1f} W")
        col3.metric("Peak Time", weather_impact['peak_generation_time'].strftime("%I:%M %p"))
        
        st.divider()
        
        st.header("Optimization Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tilt Angle Adjustment")
            box_class = "recommendation-box" if abs(tilt_recommendation['adjustment_needed']) < 5 else "warning-box"
            st.markdown(f"""
            <div class="{box_class}">
                <strong>Current Tilt:</strong> {tilt_recommendation['current_tilt']:.1f}°<br>
                <strong>Optimal Tilt:</strong> {tilt_recommendation['optimal_tilt']:.1f}°<br>
                <strong>Season:</strong> {tilt_recommendation['season']}<br>
                <strong>Expected Improvement:</strong> {tilt_recommendation['expected_improvement']}<br><br>
                <strong>Recommendation:</strong><br>
                {tilt_recommendation['recommendation']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Maintenance Status")
            for alert in maintenance_status['alerts']:
                box_class = "recommendation-box" if alert['severity'] == 'none' else "warning-box"
                st.markdown(f"""
                <div class="{box_class}">
                    <strong>{alert['type'].replace('_', ' ').title()}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        st.header("Weather Impact Analysis")
        st.markdown(f"""
        <div class="metric-card">
            <strong>Condition:</strong> {weather_impact['weather_condition']}<br>
            <strong>Avg Cloud Cover:</strong> {weather_impact['avg_cloud_cover']:.1f}%<br>
            <strong>Avg Temperature:</strong> {weather_impact['avg_temperature']:.1f}°C<br><br>
            <strong>Impact:</strong> {weather_impact['impact_description']}
        </div>
        """, unsafe_allow_html=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['cloud_cover'],
            mode='lines', name='Cloud Cover (%)', yaxis='y',
            line=dict(color='gray', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['temperature'],
            mode='lines', name='Temperature (°C)', yaxis='y2',
            line=dict(color='red', width=2)
        ))
        fig2.update_layout(
            xaxis_title="Time",
            yaxis=dict(title="Cloud Cover (%)", side='left'),
            yaxis2=dict(title="Temperature (°C)", side='right', overlaying='y'),
            hovermode='x unified', height=400, template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)


elif page == "Model Comparison":
    st.markdown('<div class="main-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analysis of All 5 Trained Models</div>', unsafe_allow_html=True)
    
    model_path = MODEL_DIR / 'best_model.pkl'
    if not model_path.exists():
        st.error("No trained model found. Please train models first using the Jupyter Notebook.")
        st.stop()
    
    try:
        model_data = joblib.load(model_path)
        best_model_name = model_data.get('model_name', 'Unknown')
        metrics = model_data.get('metrics', {})
        
        st.success(f"**Best Model:** {best_model_name} | **RMSE:** {metrics.get('RMSE', 0):.2f} W | **R²:** {metrics.get('R2', 0):.4f}")
        
    except Exception as e:
        st.error(f"Error loading model data: {e}")
        st.stop()
    
    comparison_data = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Linear Regression', 'LSTM', 'Prophet'],
        'MAE': [11.93, 12.70, 17.29, 71.32, 135.30],
        'RMSE': [24.94, 28.93, 30.44, 128.91, 140.60],
        'R2': [0.9956, 0.9940, 0.9934, 0.8845, 0.8590],
        'Training_Time_min': [4.5, 3.2, 0.5, 12.8, 6.4]
    })
    
    st.header("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_mae = px.bar(comparison_data, x='Model', y='MAE', 
                         title='Mean Absolute Error (Lower is Better)',
                         color='MAE', color_continuous_scale='RdYlGn_r')
        fig_mae.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        fig_rmse = px.bar(comparison_data, x='Model', y='RMSE',
                          title='Root Mean Squared Error (Lower is Better)',
                          color='RMSE', color_continuous_scale='RdYlGn_r')
        fig_rmse.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col3:
        fig_r2 = px.bar(comparison_data, x='Model', y='R2',
                        title='R² Score (Higher is Better)',
                        color='R2', color_continuous_scale='RdYlGn')
        fig_r2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    st.divider()

    st.header("Detailed Metrics Table")
    
    styled_df = comparison_data.style.background_gradient(
        subset=['MAE', 'RMSE'], cmap='RdYlGn_r'
    ).background_gradient(
        subset=['R2'], cmap='RdYlGn'
    ).format({
        'MAE': '{:.2f} W',
        'RMSE': '{:.2f} W',
        'R2': '{:.4f}',
        'Training_Time_min': '{:.1f} min'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.divider()
    
    st.header("Training Time vs Accuracy Trade-off")
    
    fig_scatter = px.scatter(comparison_data, x='Training_Time_min', y='R2',
                             size='RMSE', color='Model', hover_data=['MAE', 'RMSE'],
                             title='Model Efficiency: Training Time vs R² Score',
                             labels={'Training_Time_min': 'Training Time (minutes)', 'R2': 'R² Score'})
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.divider()
    

    st.header("Model Characteristics")
    
    model_info = {
        'Linear Regression': {
            'Type': 'Baseline',
            'Complexity': 'Low',
            'GPU Support': 'No',
            'Best For': 'Quick baseline, interpretability'
        },
        'Random Forest': {
            'Type': 'Ensemble',
            'Complexity': 'Medium',
            'GPU Support': 'No',
            'Best For': 'Feature importance, robustness'
        },
        'XGBoost': {
            'Type': 'Gradient Boosting',
            'Complexity': 'High',
            'GPU Support': 'Yes',
            'Best For': 'High accuracy, speed with GPU'
        },
        'LSTM': {
            'Type': 'Deep Learning',
            'Complexity': 'Very High',
            'GPU Support': 'Yes',
            'Best For': 'Temporal patterns, sequences'
        },
        'Prophet': {
            'Type': 'Time Series',
            'Complexity': 'Medium',
            'GPU Support': 'No',
            'Best For': 'Seasonality, trend analysis'
        }
    }
    
    info_df = pd.DataFrame(model_info).T
    st.dataframe(info_df, use_container_width=True)

st.divider()
st.caption("Solar Energy Forecasting System | Powered by Machine Learning")
