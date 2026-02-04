# Quick Start Guide - Solar Energy Forecasting System

## Step 1: Verify Setup ✓

The system has been created with all components:
- ✅ Data preprocessing module
- ✅ Weather API integration (with your API key)
- ✅ Computer vision tilt estimator
- ✅ Forecasting engine
- ✅ Streamlit application
- ✅ Jupyter Notebook with 5 models

## Step 2: Train Models (DO THIS NOW)

**Open the Jupyter Notebook:**

```bash
jupyter notebook notebooks/solar_forecasting_models.ipynb
```

**Then:**
1. Click "Run All" or execute each cell sequentially
2. Wait 15-30 minutes for all 5 models to train
3. The best model will be saved automatically to `models/best_model.pkl`

**Models that will be trained:**
- Linear Regression (baseline)
- Random Forest (200 trees, GPU-optimized)
- XGBoost (GPU-accelerated)
- LSTM (GPU-accelerated, 128-64 units)
- Prophet (time-series)

## Step 3: Launch the Application

**After model training completes:**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Step 4: Generate Your First Forecast

1. **Configure in sidebar:**
   - Panel: 1.6m × 1.0m, 300W
   - Location: 23.0225°N, 72.5714°E (or your location)
   - Tilt: 23° (or upload image)

2. **Click "Generate Forecast"**

3. **View results:**
   - 48-hour power forecast
   - Optimization recommendations
   - Weather impact analysis

## Troubleshooting

**If Jupyter doesn't open:**
```bash
pip install jupyter
jupyter notebook notebooks/solar_forecasting_models.ipynb
```

**If dependencies are missing:**
```bash
pip install -r requirements.txt
```

**If model file not found:**
- You must train models first (Step 2)
- Check that `models/best_model.pkl` exists

## System Features

✅ **Real-time weather** from OpenWeatherMap (your API key configured)
✅ **5 ML models** with automatic best model selection
✅ **Computer vision** for tilt angle estimation
✅ **48-hour forecasts** with confidence intervals
✅ **Actionable recommendations** for tilt, maintenance, weather

## Next Steps

1. **Train models now** (Jupyter Notebook)
2. **Test the app** (Streamlit)
3. **Try different locations** and panel configurations
4. **Upload panel images** to test tilt estimation

---

**Need help?** Check `README.md` or `walkthrough.md` for detailed documentation.
