"""
Data Preprocessing Module for Solar Energy Forecasting
Handles loading, cleaning, feature engineering, and train/test splitting
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from config import DATA_DIR, DATA_CONFIG, FEATURE_CONFIG


class SolarDataPreprocessor:
    """Preprocessor for solar generation and weather data"""
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, plant_id: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load solar generation and weather sensor data
        
        Args:
            plant_id: Plant identifier (1 or 2)
            
        Returns:
            Tuple of (generation_df, weather_df)
        """
        gen_file = DATA_DIR / f"Plant_{plant_id}_Generation_Data.csv"
        weather_file = DATA_DIR / f"Plant_{plant_id}_Weather_Sensor_Data.csv"
        
        print(f"Loading data for Plant {plant_id}...")
        gen_df = pd.read_csv(gen_file)
        weather_df = pd.read_csv(weather_file)
        
        # Parse datetime
        gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
        weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
        
        print(f"Generation data shape: {gen_df.shape}")
        print(f"Weather data shape: {weather_df.shape}")
        
        return gen_df, weather_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        print("Cleaning data...")
        initial_rows = len(df)
        
        # Check missing value ratio
        missing_ratio = df.isnull().sum() / len(df)
        if (missing_ratio > DATA_CONFIG['max_missing_ratio']).any():
            print(f"Warning: Some columns have >10% missing values")
        
        # Interpolate missing values (linear interpolation)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(
            method=DATA_CONFIG['interpolation_method'],
            limit_direction='both'
        )
        
        # Forward fill any remaining NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with NaN (should be minimal)
        df = df.dropna()
        
        print(f"Rows after cleaning: {len(df)} (removed {initial_rows - len(df)})")
        
        return df
    
    def aggregate_generation_data(self, gen_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate generation data by timestamp (sum across all inverters)
        
        Args:
            gen_df: Generation dataframe with multiple inverters
            
        Returns:
            Aggregated dataframe
        """
        print("Aggregating generation data across inverters...")
        
        agg_df = gen_df.groupby('DATE_TIME').agg({
            'DC_POWER': 'mean',
            'AC_POWER': 'mean',
            'DAILY_YIELD': 'sum',
            'TOTAL_YIELD': 'sum'
        }).reset_index()
        
        return agg_df
    
    def merge_data(self, gen_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge generation and weather data on timestamp
        
        Args:
            gen_df: Generation dataframe
            weather_df: Weather dataframe
            
        Returns:
            Merged dataframe
        """
        print("Merging generation and weather data...")
        
        # Ensure both are sorted by time
        gen_df = gen_df.sort_values('DATE_TIME')
        weather_df = weather_df.sort_values('DATE_TIME')
        
        # Merge on DATE_TIME
        merged_df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner', suffixes=('_gen', '_weather'))
        
        print(f"Merged data shape: {merged_df.shape}")
        
        return merged_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based and lag features
        
        Args:
            df: Input dataframe with DATE_TIME column
            
        Returns:
            Dataframe with engineered features
        """
        print("Engineering features...")
        
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['DATE_TIME'].dt.hour
        df['day'] = df['DATE_TIME'].dt.day
        df['month'] = df['DATE_TIME'].dt.month
        df['day_of_year'] = df['DATE_TIME'].dt.dayofyear
        df['is_weekend'] = (df['DATE_TIME'].dt.dayofweek >= 5).astype(int)
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
        
        # Lag features for AC_POWER
        for lag in FEATURE_CONFIG['lag_hours']:
            df[f'AC_POWER_lag_{lag}h'] = df['AC_POWER'].shift(lag)
        
        # Rolling statistics for IRRADIATION and temperature
        for window in FEATURE_CONFIG['rolling_windows']:
            df[f'IRRADIATION_rolling_mean_{window}h'] = df['IRRADIATION'].rolling(window=window, min_periods=1).mean()
            df[f'IRRADIATION_rolling_std_{window}h'] = df['IRRADIATION'].rolling(window=window, min_periods=1).std()
            df[f'AMBIENT_TEMPERATURE_rolling_mean_{window}h'] = df['AMBIENT_TEMPERATURE'].rolling(window=window, min_periods=1).mean()
        
        # Drop rows with NaN created by lag features
        df = df.dropna()
        
        print(f"Features after engineering: {df.shape[1]}")
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation/test split
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("Creating chronological train/validation/test split...")
        
        # Sort by time to ensure chronological order
        df = df.sort_values('DATE_TIME').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * DATA_CONFIG['train_ratio'])
        val_end = int(n * (DATA_CONFIG['train_ratio'] + DATA_CONFIG['val_ratio']))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Train set: {len(train_df)} samples ({train_df['DATE_TIME'].min()} to {train_df['DATE_TIME'].max()})")
        print(f"Validation set: {len(val_df)} samples ({val_df['DATE_TIME'].min()} to {val_df['DATE_TIME'].max()})")
        print(f"Test set: {len(test_df)} samples ({test_df['DATE_TIME'].min()} to {test_df['DATE_TIME'].max()})")
        
        return train_df, val_df, test_df
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (X, y)
        """
        # Target variable
        y = df['AC_POWER']
        
        # Features to exclude (all non-numeric and target-related columns)
        exclude_cols = ['DATE_TIME', 'AC_POWER', 'DC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']
        
        # Also exclude any columns with 'PLANT_ID' or 'SOURCE_KEY' in the name
        exclude_cols.extend([col for col in df.columns if 'PLANT_ID' in col or 'SOURCE_KEY' in col])
        
        # Select only numeric columns that are not in exclude list
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[feature_cols]
        
        self.feature_names = feature_cols
        
        print(f"Selected {len(feature_cols)} numeric features for training")
        
        return X, y
    
    def preprocess_pipeline(self, plant_id: int = 1) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            plant_id: Plant identifier
            
        Returns:
            Dictionary containing processed data
        """
        print("="*60)
        print("SOLAR DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        gen_df, weather_df = self.load_data(plant_id)
        
        # Aggregate generation data
        gen_df = self.aggregate_generation_data(gen_df)
        
        # Clean data
        gen_df = self.clean_data(gen_df)
        weather_df = self.clean_data(weather_df)
        
        # Merge datasets
        merged_df = self.merge_data(gen_df, weather_df)
        
        # Engineer features
        featured_df = self.engineer_features(merged_df)
        
        # Create train/val/test split
        train_df, val_df, test_df = self.create_train_test_split(featured_df)
        
        # Prepare features and target
        X_train, y_train = self.prepare_features_target(train_df)
        X_val, y_val = self.prepare_features_target(val_df)
        X_test, y_test = self.prepare_features_target(test_df)
        
        print("="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_dates': train_df['DATE_TIME'],
            'val_dates': val_df['DATE_TIME'],
            'test_dates': test_df['DATE_TIME'],
            'feature_names': self.feature_names
        }


if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = SolarDataPreprocessor()
    data = preprocessor.preprocess_pipeline(plant_id=1)
    
    print("\nData shapes:")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"\nFeatures ({len(data['feature_names'])}):")
    print(data['feature_names'])
