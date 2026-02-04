"""
Forecasting Engine Module
Generates forecasts and optimization recommendations
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_DIR, FORECAST_CONFIG


class ForecastingEngine:
    """Generate solar power forecasts and recommendations"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize forecasting engine
        
        Args:
            model_path: Path to trained model file (default: best_model.pkl)
        """
        if model_path is None:
            model_path = MODEL_DIR / "best_model.pkl"
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"Warning: Model file not found at {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and scaler"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names')
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def generate_forecast(
        self, 
        weather_forecast: List[Dict],
        panel_specs: Dict,
        hours: int = 48
    ) -> pd.DataFrame:
        """
        Generate power forecast for next N hours
        
        Args:
            weather_forecast: List of weather forecast dictionaries
            panel_specs: Panel specifications (capacity, tilt, etc.)
            hours: Number of hours to forecast
            
        Returns:
            DataFrame with forecast
        """
        if self.model is None:
            raise ValueError("No model loaded. Cannot generate forecast.")
        
        # Prepare forecast features
        forecast_df = self._prepare_forecast_features(weather_forecast, panel_specs)
        
        # Generate predictions
        if self.scaler:
            X_scaled = self.scaler.transform(forecast_df[self.feature_names])
            predictions = self.model.predict(X_scaled)
        else:
            predictions = self.model.predict(forecast_df[self.feature_names])
        
        # Create forecast dataframe
        forecast_result = pd.DataFrame({
            'timestamp': forecast_df['timestamp'],
            'predicted_power': np.maximum(predictions, 0),  # Ensure non-negative
            'temperature': forecast_df['temperature'],
            'cloud_cover': forecast_df['cloud_cover'],
            'irradiation_estimate': forecast_df['irradiation_estimate']
        })
        
        # Add confidence intervals
        forecast_result = self._add_confidence_intervals(forecast_result, predictions)
        
        return forecast_result
    
    def _prepare_forecast_features(
        self, 
        weather_forecast: List[Dict],
        panel_specs: Dict
    ) -> pd.DataFrame:
        """Prepare features from weather forecast"""
        
        forecast_data = []
        for weather in weather_forecast:
            timestamp = pd.to_datetime(weather['timestamp'])
            
            # Estimate irradiation from cloud cover (simplified model)
            cloud_cover = weather['cloud_cover']
            hour = timestamp.hour
            
            # Clear sky irradiation (simplified)
            if 6 <= hour <= 18:
                clear_sky_irr = 0.8 * np.sin((hour - 6) * np.pi / 12)
            else:
                clear_sky_irr = 0.0
            
            # Adjust for cloud cover
            irradiation = clear_sky_irr * (1 - cloud_cover / 100)
            
            features = {
                'timestamp': timestamp,
                'AMBIENT_TEMPERATURE': weather['temperature'],
                'MODULE_TEMPERATURE': weather['temperature'] + 10,  # Estimate
                'IRRADIATION': irradiation,
                'cloud_cover': cloud_cover,
                'irradiation_estimate': irradiation,
                'temperature': weather['temperature'],
                'humidity': weather.get('humidity', 50),
                'wind_speed': weather.get('wind_speed', 3),
                'hour': timestamp.hour,
                'day': timestamp.day,
                'month': timestamp.month,
                'day_of_year': timestamp.dayofyear,
                'is_weekend': int(timestamp.dayofweek >= 5),
                'season': (timestamp.month % 12 + 3) // 3
            }
            
            forecast_data.append(features)
        
        df = pd.DataFrame(forecast_data)
        
        # Add lag and rolling features (use last known values or estimates)
        for lag in [1, 3, 6, 24]:
            df[f'AC_POWER_lag_{lag}h'] = 0  # Will be updated with recent data
        
        for window in [3, 6, 24]:
            df[f'IRRADIATION_rolling_mean_{window}h'] = df['IRRADIATION'].rolling(window, min_periods=1).mean()
            df[f'IRRADIATION_rolling_std_{window}h'] = df['IRRADIATION'].rolling(window, min_periods=1).std().fillna(0)
            df[f'AMBIENT_TEMPERATURE_rolling_mean_{window}h'] = df['AMBIENT_TEMPERATURE'].rolling(window, min_periods=1).mean()
        
        return df
    
    def _add_confidence_intervals(
        self, 
        forecast_df: pd.DataFrame, 
        predictions: np.ndarray
    ) -> pd.DataFrame:
        """Add confidence intervals to forecast"""
        
        # Simple confidence interval based on prediction uncertainty
        # In production, use quantile regression or bootstrap
        std_error = np.std(predictions) * 0.2  # Simplified
        
        forecast_df['lower_bound'] = np.maximum(
            forecast_df['predicted_power'] - 1.96 * std_error, 0
        )
        forecast_df['upper_bound'] = forecast_df['predicted_power'] + 1.96 * std_error
        
        return forecast_df
    
    def recommend_tilt_adjustment(
        self, 
        current_tilt: float,
        latitude: float,
        season: int
    ) -> Dict:
        """
        Recommend optimal tilt angle adjustment
        
        Args:
            current_tilt: Current tilt angle in degrees
            latitude: Geographic latitude
            season: Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
            
        Returns:
            Dictionary with recommendation
        """
        # Optimal tilt varies by season
        seasonal_adjustment = {
            1: 15,   # Winter: steeper angle
            2: 0,    # Spring: near latitude
            3: -15,  # Summer: shallower angle
            4: 0     # Fall: near latitude
        }
        
        optimal_tilt = abs(latitude) + seasonal_adjustment.get(season, 0)
        optimal_tilt = np.clip(optimal_tilt, 0, 90)
        
        adjustment = optimal_tilt - current_tilt
        
        # Calculate expected improvement
        if abs(adjustment) < 5:
            improvement = "minimal"
            recommendation = f"Current tilt angle ({current_tilt:.1f}°) is near optimal."
        elif abs(adjustment) < 15:
            improvement = "moderate (2-5% gain)"
            recommendation = f"Adjust tilt to {optimal_tilt:.1f}° for {improvement} in energy capture."
        else:
            improvement = "significant (5-10% gain)"
            recommendation = f"Adjust tilt to {optimal_tilt:.1f}° for {improvement} in energy capture."
        
        return {
            'current_tilt': current_tilt,
            'optimal_tilt': optimal_tilt,
            'adjustment_needed': adjustment,
            'expected_improvement': improvement,
            'recommendation': recommendation,
            'season': ['Winter', 'Spring', 'Summer', 'Fall'][season - 1]
        }
    
    def detect_maintenance_needs(
        self, 
        forecast_df: pd.DataFrame,
        panel_specs: Dict
    ) -> Dict:
        """
        Detect potential maintenance needs
        
        Args:
            forecast_df: Forecast dataframe
            panel_specs: Panel specifications
            
        Returns:
            Dictionary with maintenance alerts
        """
        alerts = []
        
        # Calculate expected clear-sky performance
        rated_capacity = panel_specs.get('rated_capacity', 300)
        
        # Check for efficiency drops
        avg_predicted = forecast_df['predicted_power'].mean()
        
        # During daylight hours (simplified)
        daylight_forecast = forecast_df[
            (forecast_df['timestamp'].dt.hour >= 8) & 
            (forecast_df['timestamp'].dt.hour <= 16)
        ]
        
        if len(daylight_forecast) > 0:
            avg_daylight_power = daylight_forecast['predicted_power'].mean()
            efficiency = avg_daylight_power / rated_capacity
            
            if efficiency < FORECAST_CONFIG['clear_sky_efficiency_threshold']:
                alerts.append({
                    'type': 'efficiency_warning',
                    'message': f"Predicted efficiency ({efficiency*100:.1f}%) below threshold. Consider panel cleaning or inspection.",
                    'severity': 'medium'
                })
        
        # Check for unusual patterns
        power_std = forecast_df['predicted_power'].std()
        if power_std < 10:  # Very low variance
            alerts.append({
                'type': 'pattern_anomaly',
                'message': "Unusual power generation pattern detected. Verify system operation.",
                'severity': 'low'
            })
        
        if not alerts:
            alerts.append({
                'type': 'normal',
                'message': "System operating within normal parameters.",
                'severity': 'none'
            })
        
        return {
            'alerts': alerts,
            'maintenance_recommended': any(a['severity'] in ['high', 'medium'] for a in alerts)
        }
    
    def analyze_weather_impact(
        self, 
        forecast_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze weather impact on generation
        
        Args:
            forecast_df: Forecast dataframe
            
        Returns:
            Dictionary with weather impact analysis
        """
        # Calculate impact of cloud cover
        avg_cloud_cover = forecast_df['cloud_cover'].mean()
        
        if avg_cloud_cover < 20:
            weather_condition = "Excellent"
            impact = "Minimal impact expected. Optimal generation conditions."
        elif avg_cloud_cover < 50:
            weather_condition = "Good"
            impact = "Slight reduction due to partial cloud cover."
        elif avg_cloud_cover < 80:
            weather_condition = "Fair"
            impact = "Moderate reduction (20-40%) due to cloudy conditions."
        else:
            weather_condition = "Poor"
            impact = "Significant reduction (40-60%) due to heavy cloud cover."
        
        # Find peak generation time
        peak_idx = forecast_df['predicted_power'].idxmax()
        peak_time = forecast_df.loc[peak_idx, 'timestamp']
        peak_power = forecast_df.loc[peak_idx, 'predicted_power']
        
        # Calculate total expected energy
        total_energy = forecast_df['predicted_power'].sum() / 1000  # kWh (assuming hourly data)
        
        return {
            'weather_condition': weather_condition,
            'avg_cloud_cover': avg_cloud_cover,
            'impact_description': impact,
            'peak_generation_time': peak_time,
            'peak_power': peak_power,
            'total_expected_energy_kwh': total_energy,
            'avg_temperature': forecast_df['temperature'].mean()
        }


if __name__ == "__main__":
    print("Forecasting Engine Module")
    print("Note: Requires trained model to generate forecasts")
    print("Example usage:")
    print("  engine = ForecastingEngine()")
    print("  forecast = engine.generate_forecast(weather_data, panel_specs)")
