import pandas as pd
import numpy as np

def load_and_align_oil(oil_path):
    """
    Loads oil prices and fills missing values (e.g. weekends) using linear interpolation.
    Also creates rolling oil price features.
    """
    oil = pd.read_csv(oil_path)
    oil['date'] = pd.to_datetime(oil['date'])
    
    # Generate full date range to ensure missing dates are filled
    full_dates = pd.date_range(start=oil['date'].min(), end=oil['date'].max())
    oil = oil.set_index('date').reindex(full_dates)
    oil.index.name = 'date'
    
    # Linearly interpolate missing oil prices
    oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='linear', limit_direction='both')
    
    # Rolling oil price statistics to capture trends
    oil['oil_roll_mean_3'] = oil['dcoilwtico'].rolling(3).mean()
    oil['oil_roll_mean_7'] = oil['dcoilwtico'].rolling(7).mean()
    oil['oil_roll_std_7'] = oil['dcoilwtico'].rolling(7).std()
    
    # Fill remaining NaNs from rolling window starts
    oil = oil.bfill().reset_index()
    return oil

def map_holidays(df, holidays_path, stores):
    """
    Maps holidays to each store-date combination.
    Holidays can be:
      - National: Applies to all stores.
      - Regional: Applies to stores in the same state.
      - Local: Applies to stores in the same city.
    Excludes transferred holidays (transferred == True).
    """
    holidays = pd.read_csv(holidays_path)
    holidays['date'] = pd.to_datetime(holidays['date'])
    
    # Filter out transferred holidays
    holidays = holidays[holidays['transferred'] == False]
    
    # Separate holidays by locale
    national = holidays[holidays['locale'] == 'National']
    regional = holidays[holidays['locale'] == 'Regional']
    local = holidays[holidays['locale'] == 'Local']
    
    # Create target columns in df
    df['is_holiday'] = 0
    df['holiday_type'] = 'None'
    
    # 1. Map National Holidays (apply to all store numbers)
    for _, row in national.iterrows():
        mask = df['date'] == row['date']
        df.loc[mask, 'is_holiday'] = 1
        df.loc[mask, 'holiday_type'] = row['type']
        
    # 2. Map Regional Holidays (match on state)
    for _, row in regional.iterrows():
        # Find stores in that state
        matching_stores = stores[stores['state'] == row['locale_name']]['store_nbr'].unique()
        mask = (df['date'] == row['date']) & (df['store_nbr'].isin(matching_stores))
        df.loc[mask, 'is_holiday'] = 1
        df.loc[mask, 'holiday_type'] = row['type']
        
    # 3. Map Local Holidays (match on city)
    for _, row in local.iterrows():
        # Find stores in that city
        matching_stores = stores[stores['city'] == row['locale_name']]['store_nbr'].unique()
        mask = (df['date'] == row['date']) & (df['store_nbr'].isin(matching_stores))
        df.loc[mask, 'is_holiday'] = 1
        df.loc[mask, 'holiday_type'] = row['type']
        
    return df

def build_features(train_path, test_path, stores_path, oil_path, holidays_path, transactions_path=None):
    """
    Main feature engineering pipeline. Merges tables, extracts calendar features,
    and generates non-leaking lag/rolling window statistics.
    """
    # Load primary datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    stores = pd.read_csv(stores_path)
    
    # Align date columns
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])
    
    # Combine train and test to compute lags and rolling features across the boundary
    # Test target 'sales' is filled with NaN
    test['sales'] = np.nan
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df = df.sort_values(by=['store_nbr', 'family', 'date']).reset_index(drop=True)
    
    # Merge store metadata
    df = df.merge(stores, on='store_nbr', how='left')
    
    # Load and merge oil prices
    oil = load_and_align_oil(oil_path)
    df = df.merge(oil, on='date', how='left')
    
    # Map holidays
    df = map_holidays(df, holidays_path, stores)
    
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Log transform target for model training (RMSLE optimization)
    # The original target is 'sales'. We use log1p to stabilize variance.
    df['log_sales'] = np.log1p(df['sales'])
    
    # Generate Lag Features
    # The forecast window is 16 days. To prevent data leakage, any lag used
    # in the feature set must be at least 16 days back.
    lags = [16, 17, 18, 19, 20, 21, 22, 28, 35]
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['log_sales'].shift(lag)
        
    # Generate Rolling Statistics
    # We calculate rolling averages of the 16-day lag feature to capture trends
    # without leaking future sales information.
    rolling_windows = [7, 14, 28]
    for window in rolling_windows:
        df[f'sales_roll_mean_{window}'] = df.groupby(['store_nbr', 'family'])['sales_lag_16'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'sales_roll_std_{window}'] = df.groupby(['store_nbr', 'family'])['sales_lag_16'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        
    # Fill rolling std NaNs (which happen at window sizes of 1) with 0
    std_cols = [c for c in df.columns if 'roll_std' in c]
    df[std_cols] = df[std_cols].fillna(0)
    
    # Categorical Encodings (Convert strings to category types for LightGBM/CatBoost)
    categorical_cols = ['family', 'city', 'state', 'type', 'holiday_type']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        
    # Separate back into train and test
    train_feat = df[df['sales'].notnull()].copy()
    test_feat = df[df['sales'].isnull()].copy()
    
    # Drop rows at the very beginning of train that have NaN lag values due to shifting
    # Max lag is 35, plus 28 for rolling window means we need to drop the first 63 days per series
    train_feat = train_feat.dropna(subset=['sales_lag_35']).reset_index(drop=True)
    
    return train_feat, test_feat
