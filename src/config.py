
import os
from pathlib import Path

def load_env_file():
    """Load .env file manually"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Dataset"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
NOTEBOOK_DIR = BASE_DIR / "notebooks"

MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

API_RATE_LIMIT = 60 
CACHE_EXPIRY_MINUTES = 30

MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,  # Increased from 100
        "max_depth": 20,  # Increased from 15
        "n_jobs": -1,  # Use all CPU cores
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 500,  # Increased for better performance
        "max_depth": 10,
        "learning_rate": 0.05,
        "tree_method": "gpu_hist",  # GPU acceleration
        "gpu_id": 0,
        "random_state": 42
    },
    "lstm": {
        "sequence_length": 24,
        "lstm_units": [128, 64],  # Increased from [64, 32]
        "dropout": 0.2,
        "batch_size": 128,  # Increased from 32
        "epochs": 100,  # Increased from 50
        "early_stopping_patience": 15,
        "use_gpu": True
    },
    "prophet": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10,
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": True
    }
}

DATA_CONFIG = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "interpolation_method": "linear",
    "max_missing_ratio": 0.1  # Max 10% missing values allowed
}

FEATURE_CONFIG = {
    "lag_hours": [1, 3, 6, 24],
    "rolling_windows": [3, 6, 24],
    "time_features": ["hour", "day", "month", "season", "is_weekend"]
}

CV_CONFIG = {
    "canny_threshold1": 50,
    "canny_threshold2": 150,
    "min_angle": 0,
    "max_angle": 90,
    "default_angle_latitude_based": True
}

FORECAST_CONFIG = {
    "forecast_hours": 48,
    "confidence_level": 0.95,
    "clear_sky_efficiency_threshold": 0.85
}

UI_CONFIG = {
    "page_title": "Solar Energy Forecasting System",
    "page_icon": "☀️",
    "layout": "wide",
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6"
    }
}
