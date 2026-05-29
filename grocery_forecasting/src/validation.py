import pandas as pd

def get_time_series_split(df, val_start_date='2017-08-01', val_end_date='2017-08-15'):
    """
    Splits the dataset into train and validation sets chronologically.
    Prevents future data leakage by ensuring no validation dates are present in the training set.
    
    Parameters:
      df: Preprocessed DataFrame containing a 'date' column of datetime type.
      val_start_date: Start date of the validation window (string YYYY-MM-DD).
      val_end_date: End date of the validation window (string YYYY-MM-DD).
      
    Returns:
      train_df: Chronological training set.
      val_df: Chronological validation set matching the length and day-of-week structure of the Kaggle test set.
    """
    val_start = pd.to_datetime(val_start_date)
    val_end = pd.to_datetime(val_end_date)
    
    # Validation mask
    val_mask = (df['date'] >= val_start) & (df['date'] <= val_end)
    
    # Training mask (all data strictly before the start of the validation window)
    train_mask = df['date'] < val_start
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    
    print(f"Validation Split Details:")
    print(f"  Training Range:   {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} records)")
    print(f"  Validation Range: {val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')} ({len(val_df)} records)")
    
    # Sanity checks
    assert train_df['date'].max() < val_df['date'].min(), "CRITICAL ERROR: Time leakage detected! Training data overlaps with validation data."
    
    return train_df, val_df
