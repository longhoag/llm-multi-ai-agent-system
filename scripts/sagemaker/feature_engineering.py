"""
SageMaker Processing Script for Advanced Feature Engineering
Production-grade feature engineering for stock prediction pipeline
"""

import json
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def create_advanced_features(df):
    """Create advanced technical indicators and features"""
    
    # Advanced moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
    
    # Multiple timeframe RSI
    for period in [7, 14, 21, 30]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD variations
    for fast, slow, signal in [(12, 26, 9), (8, 21, 5), (5, 13, 8)]:
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        df[f'MACD_{fast}_{slow}'] = macd
        df[f'MACD_Signal_{fast}_{slow}_{signal}'] = macd_signal
        df[f'MACD_Hist_{fast}_{slow}_{signal}'] = macd - macd_signal
    
    # Bollinger Bands variations
    for period, std_dev in [(20, 2), (20, 2.5), (10, 1.5)]:
        bb_middle = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        df[f'BB_Upper_{period}_{std_dev}'] = bb_middle + (bb_std * std_dev)
        df[f'BB_Lower_{period}_{std_dev}'] = bb_middle - (bb_std * std_dev)
        df[f'BB_Width_{period}_{std_dev}'] = (df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}']) / bb_middle
        df[f'BB_Position_{period}_{std_dev}'] = (df['Close'] - df[f'BB_Lower_{period}_{std_dev}']) / (df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}'])
    
    # Advanced price features
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['Gap'] = df['Open'] - df['Close'].shift(1)
    df['Gap_Percent'] = df['Gap'] / df['Close'].shift(1)
    
    # Volatility measures
    for period in [5, 10, 20, 30]:
        returns = df['Close'].pct_change()
        df[f'Volatility_{period}'] = returns.rolling(window=period).std()
        df[f'ATR_{period}'] = ((df['High'] - df['Low']).rolling(window=period).mean())
    
    # Volume features
    for period in [5, 10, 20]:
        df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
        df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_SMA_{period}']
        df[f'Price_Volume_{period}'] = df['Close'] * df['Volume']
        df[f'VWAP_{period}'] = (df[f'Price_Volume_{period}'].rolling(window=period).sum() / 
                               df['Volume'].rolling(window=period).sum())
    
    # Momentum indicators
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period)
        df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
    
    # Support and Resistance levels
    for period in [20, 50]:
        df[f'Resistance_{period}'] = df['High'].rolling(window=period).max()
        df[f'Support_{period}'] = df['Low'].rolling(window=period).min()
        df[f'Support_Resistance_Ratio_{period}'] = (df['Close'] - df[f'Support_{period}']) / (df[f'Resistance_{period}'] - df[f'Support_{period}'])
    
    # Trend indicators
    for period in [10, 20, 50]:
        df[f'Trend_{period}'] = df['Close'] / df['Close'].rolling(window=period).mean()
        df[f'Upper_Shadow_{period}'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df[f'Lower_Shadow_{period}'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
    
    return df

def process_stock_data():
    """Main processing function for SageMaker"""
    
    # Input and output paths
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'
    
    print("Starting SageMaker feature engineering...")
    
    # Find input files
    input_files = list(Path(input_path).glob('*.json'))
    if not input_files:
        raise ValueError("No JSON files found in input path")
    
    processed_files = []
    
    for input_file in input_files:
        print(f"Processing file: {input_file}")
        
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract time series
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            print(f"No time series data in {input_file}")
            continue
        
        # Convert to DataFrame
        df_data = []
        for date, values in time_series.items():
            row = {
                'Date': pd.to_datetime(date),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data).sort_values('Date')
        print(f"Loaded {len(df)} records")
        
        # Create advanced features
        df = create_advanced_features(df)
        
        # Remove NaN values
        df_clean = df.dropna()
        print(f"After feature engineering: {len(df_clean)} records with {len(df_clean.columns)} features")
        
        # Prepare output data
        symbol = data.get('Meta Data', {}).get('2. Symbol', 'Unknown')
        
        processed_data = {
            'Meta Data': {
                'Symbol': symbol,
                'Processing_Type': 'SageMaker Advanced Feature Engineering',
                'Processing_Date': pd.Timestamp.now().isoformat(),
                'Features_Created': [col for col in df_clean.columns if col != 'Date'],
                'Records_Count': len(df_clean),
                'Original_Records': len(df),
                'Features_Count': len(df_clean.columns) - 1  # Exclude Date
            },
            'Processed_Data': df_clean.to_dict('records')
        }
        
        # Save processed data
        output_file = Path(output_path) / f"{symbol}_advanced_features.json"
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        processed_files.append(str(output_file))
        print(f"Saved processed data to: {output_file}")
    
    # Create processing summary
    summary = {
        'processing_summary': {
            'files_processed': len(processed_files),
            'output_files': processed_files,
            'processing_date': pd.Timestamp.now().isoformat(),
            'processor': 'SageMaker Advanced Feature Engineering'
        }
    }
    
    summary_file = Path(output_path) / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Processing complete. Summary saved to: {summary_file}")

if __name__ == "__main__":
    process_stock_data()
