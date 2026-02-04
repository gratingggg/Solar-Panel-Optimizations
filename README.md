# Solar Energy Forecasting and Optimization System

AI-powered solar energy forecasting system for real-world solar operations. Provides 48-hour power generation forecasts, optimization recommendations, and maintenance alerts.

## Features

- **Five ML Models**: Linear Regression, Random Forest, XGBoost (GPU), LSTM (GPU), Prophet
- **Real-time Weather Integration**: OpenWeatherMap API with intelligent caching
- **Computer Vision**: Tilt angle estimation from uploaded panel images
- **48-Hour Forecasts**: With 95% confidence intervals
- **Optimization Insights**: Tilt adjustments, maintenance alerts, weather impact analysis
- **Professional UI**: Clean Streamlit interface with interactive visualizations

## Hardware Optimizations

Configured for high-performance systems:
- **XGBoost**: GPU-accelerated training (`gpu_hist`)
- **LSTM**: Larger units (128, 64), increased batch size (128)
- **Random Forest**: 200 trees, max_depth=20, parallel processing
- **Memory**: Optimized for 32GB RAM

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

Open and run all cells in the Jupyter Notebook:

```bash
jupyter notebook notebooks/solar_forecasting_models.ipynb
```

This will:
- Load and preprocess the solar generation data
- Train all 5 models
- Evaluate and compare performance
- Save the best model to `models/best_model.pkl`

**Estimated training time**: 15-30 minutes (with GPU)

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. **Configure Panel Specifications**:
   - Enter panel dimensions (width, height)
   - Specify rated capacity in Watts
   - Set geographic coordinates

2. **Set Tilt Angle**:
   - Manual input, or
   - Upload panel image for automatic estimation

3. **Generate Forecast**:
   - Click "Generate Forecast"
   - View 48-hour power predictions
   - Review optimization recommendations

## Project Structure

```
d:/Solar Panel/
├── Dataset/                    # Solar generation and weather data
├── notebooks/
│   └── solar_forecasting_models.ipynb  # Model training notebook
├── src/
│   ├── config.py              # Configuration settings
│   ├── data_preprocessing.py  # Data loading and feature engineering
│   ├── weather_api.py         # Weather API client with caching
│   ├── tilt_angle_estimator.py  # Computer vision module
│   └── forecasting_engine.py  # Forecast generation and recommendations
├── models/
│   └── best_model.pkl         # Trained model (created after training)
├── cache/                     # Weather API cache
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
└── .env                       # API keys (already configured)
```

## Model Performance

The system trains and evaluates 5 models:

| Model | Use Case | GPU Support |
|-------|----------|-------------|
| Linear Regression | Baseline | No |
| Random Forest | Feature importance | No |
| XGBoost | High accuracy | Yes |
| LSTM | Time-series patterns | Yes |
| Prophet | Seasonality handling | No |

Best model is automatically selected based on lowest RMSE.

## API Configuration

OpenWeatherMap API key is already configured in `.env` file.

**Rate Limits**: 60 calls/minute (free tier)
**Caching**: 30-minute expiry to minimize API usage

## Optimization Recommendations

The system provides:

1. **Tilt Angle Adjustments**:
   - Seasonal recommendations
   - Latitude-based optimization
   - Expected performance improvement

2. **Maintenance Alerts**:
   - Efficiency drop detection
   - Pattern anomaly identification
   - System health status

3. **Weather Impact Analysis**:
   - Cloud cover effects
   - Temperature impact
   - Peak generation timing

## Data Requirements

The system uses the provided dataset:
- `Plant_1_Generation_Data.csv` (68,780 records)
- `Plant_1_Weather_Sensor_Data.csv` (3,184 records)

Data includes:
- AC/DC power output
- Ambient and module temperature
- Solar irradiation
- 15-minute intervals

## Technical Details

**Preprocessing**:
- Interpolation for missing values (no row deletion)
- Chronological train/val/test split (70/15/15)
- Feature engineering: lag features, rolling statistics, time features

**Evaluation Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

**Forecast Output**:
- Hourly predictions for 48 hours
- Confidence intervals (95%)
- Total expected energy (kWh)
- Peak power and timing

## Troubleshooting

**Model not found error**:
- Run the Jupyter Notebook to train models first
- Ensure `models/best_model.pkl` exists

**API errors**:
- Check internet connection
- Verify API key in `.env` file
- System will use fallback mode if API unavailable

**GPU not detected**:
- Install CUDA-enabled TensorFlow: `pip install tensorflow-gpu`
- XGBoost will automatically fall back to CPU if GPU unavailable

## License

This project is for educational and operational use in solar energy forecasting.

## Support

For issues or questions, refer to the implementation plan in:
`C:\Users\agrer\.gemini\antigravity\brain\59f79fb9-9415-4744-a4d4-76fbf3af9153\implementation_plan.md`
