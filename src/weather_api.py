"""
Weather API Integration Module
Handles OpenWeatherMap API calls with intelligent caching
"""
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

from config import (
    OPENWEATHER_API_KEY, 
    OPENWEATHER_BASE_URL,
    CACHE_DIR,
    API_RATE_LIMIT,
    CACHE_EXPIRY_MINUTES
)


class WeatherAPIClient:
    """Client for OpenWeatherMap API with caching"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.base_url = OPENWEATHER_BASE_URL
        self.cache_file = CACHE_DIR / "weather_cache.json"
        self.rate_limit = API_RATE_LIMIT
        self.last_call_time = 0
        
        if not self.api_key:
            print("Warning: No API key provided. Using fallback mode.")
    
    def _rate_limit_check(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        min_interval = 60.0 / self.rate_limit  # seconds between calls
        
        if time_since_last_call < min_interval:
            sleep_time = min_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _load_cache(self) -> Dict:
        """Load cache from file"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self, cache: Dict):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    
    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Get cached data if still fresh"""
        cache = self._load_cache()
        
        if cache_key in cache:
            cached_data = cache[cache_key]
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            
            # Check if cache is still fresh
            if datetime.now() - cached_time < timedelta(minutes=CACHE_EXPIRY_MINUTES):
                print(f"Using cached data (age: {(datetime.now() - cached_time).seconds // 60} minutes)")
                return cached_data['data']
        
        return None
    
    def _cache_response(self, cache_key: str, data: Dict):
        """Cache API response"""
        cache = self._load_cache()
        cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self._save_cache(cache)
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get current weather data
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather data dictionary or None if failed
        """
        cache_key = f"current_{lat}_{lon}"
        
        # Check cache first
        cached_data = self._get_cached(cache_key)
        if cached_data:
            return cached_data
        
        if not self.api_key:
            return self._get_fallback_weather()
        
        # Make API call
        try:
            self._rate_limit_check()
            
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant fields
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'cloud_cover': data['clouds']['all'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the response
            self._cache_response(cache_key, weather_data)
            
            return weather_data
            
        except Exception as e:
            print(f"Error fetching current weather: {e}")
            return self._get_fallback_weather()
    
    def get_forecast(self, lat: float, lon: float, hours: int = 48) -> Optional[List[Dict]]:
        """
        Get weather forecast
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Number of hours to forecast (default 48)
            
        Returns:
            List of forecast data dictionaries or None if failed
        """
        cache_key = f"forecast_{lat}_{lon}_{hours}"
        
        # Check cache first
        cached_data = self._get_cached(cache_key)
        if cached_data:
            return cached_data
        
        if not self.api_key:
            return self._get_fallback_forecast(hours)
        
        # Make API call
        try:
            self._rate_limit_check()
            
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(hours // 3, 40)  # API returns 3-hour intervals, max 40 periods
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract forecast data
            forecast_list = []
            for item in data['list']:
                forecast_data = {
                    'timestamp': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'cloud_cover': item['clouds']['all'],
                    'wind_speed': item['wind']['speed'],
                    'description': item['weather'][0]['description']
                }
                forecast_list.append(forecast_data)
            
            # Cache the response
            self._cache_response(cache_key, forecast_list)
            
            return forecast_list
            
        except Exception as e:
            print(f"Error fetching forecast: {e}")
            return self._get_fallback_forecast(hours)
    
    def _get_fallback_weather(self) -> Dict:
        """Return fallback weather data based on historical averages"""
        print("Using fallback weather data (historical averages)")
        return {
            'temperature': 28.0,  # Average temperature from dataset
            'humidity': 45.0,
            'pressure': 1013.0,
            'cloud_cover': 30.0,
            'wind_speed': 3.5,
            'description': 'clear sky (fallback)',
            'timestamp': datetime.now().isoformat(),
            'is_fallback': True
        }
    
    def _get_fallback_forecast(self, hours: int) -> List[Dict]:
        """Return fallback forecast based on historical patterns"""
        print(f"Using fallback forecast data for {hours} hours")
        forecast = []
        
        current_time = datetime.now()
        for i in range(0, hours, 3):  # 3-hour intervals
            forecast_time = current_time + timedelta(hours=i)
            hour = forecast_time.hour
            
            # Simple pattern: warmer during day, cooler at night
            if 6 <= hour <= 18:
                temp = 28 + (hour - 12) * 2  # Peak at noon
                cloud = 20
            else:
                temp = 22 - abs(hour - 3) * 0.5
                cloud = 10
            
            forecast.append({
                'timestamp': forecast_time.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': temp,
                'humidity': 45.0,
                'pressure': 1013.0,
                'cloud_cover': cloud,
                'wind_speed': 3.5,
                'description': 'clear sky (fallback)',
                'is_fallback': True
            })
        
        return forecast


if __name__ == "__main__":
    # Test the API client
    client = WeatherAPIClient()
    
    # Test coordinates (Ahmedabad, India - near dataset location)
    lat, lon = 23.0225, 72.5714
    
    print("Testing current weather...")
    current = client.get_current_weather(lat, lon)
    if current:
        print(f"Temperature: {current['temperature']}°C")
        print(f"Humidity: {current['humidity']}%")
        print(f"Cloud Cover: {current['cloud_cover']}%")
    
    print("\nTesting forecast...")
    forecast = client.get_forecast(lat, lon, hours=48)
    if forecast:
        print(f"Forecast periods: {len(forecast)}")
        print(f"First period: {forecast[0]}")
