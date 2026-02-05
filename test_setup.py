
import sys
sys.path.append('src')

print("="*60)
print("SOLAR FORECASTING SYSTEM - SETUP VERIFICATION")
print("="*60)

print("\n1. Testing imports...")
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost as xgb
    import streamlit as st
    print("   ✓ Core libraries imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

print("\n2. Testing custom modules...")
try:
    from config import DATA_DIR, MODEL_DIR
    from data_preprocessing import SolarDataPreprocessor
    from weather_api import WeatherAPIClient
    from tilt_angle_estimator import TiltAngleEstimator
    from forecasting_engine import ForecastingEngine
    print("   ✓ Custom modules imported successfully")
except ImportError as e:
    print(f"   ✗ Module import error: {e}")
    sys.exit(1)

print("\n3. Checking data directory...")
if DATA_DIR.exists():
    files = list(DATA_DIR.glob("*.csv"))
    print(f"   ✓ Data directory found with {len(files)} CSV files")
else:
    print(f"   ✗ Data directory not found: {DATA_DIR}")


print("\n4. Testing Weather API client...")
try:
    client = WeatherAPIClient()
    if client.api_key:
        print(f"   ✓ API key configured")
    else:
        print("   ⚠ No API key (will use fallback mode)")
except Exception as e:
    print(f"   ✗ API client error: {e}")

print("\n5. Testing data preprocessing...")
try:
    preprocessor = SolarDataPreprocessor()
    print("   ✓ Preprocessor initialized")
except Exception as e:
    print(f"   ✗ Preprocessor error: {e}")

print("\n" + "="*60)
print("SETUP VERIFICATION COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Run: jupyter notebook notebooks/solar_forecasting_models.ipynb")
print("2. Train all models (15-30 minutes)")
print("3. Run: streamlit run app.py")
print("="*60)
