import os
import argparse
import numpy as np
import pandas as pd

# Define a fallback model in case LightGBM is not installed (offline sandbox safety)
class FallbackLinearRegression:
    """
    A simple baseline regression model using NumPy.
    Used as an offline fallback when LightGBM is not available.
    """
    def __init__(self):
        self.weights = None
        self.mean = None
        self.std = None
        
    def _preprocess(self, X):
        # Convert categoricals to numeric codes
        X_num = X.copy()
        for col in X_num.columns:
            if X_num[col].dtype.name == 'category':
                X_num[col] = X_num[col].cat.codes
        # Fill missing values
        X_num = X_num.fillna(0)
        return X_num.values
        
    def fit(self, X, y):
        X_mat = self._preprocess(X)
        # Add bias term
        X_mat = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])
        # Standardize for stability
        self.mean = np.mean(X_mat[:, 1:], axis=0)
        self.std = np.std(X_mat[:, 1:], axis=0)
        self.std[self.std == 0] = 1.0 # Avoid division by zero
        X_mat[:, 1:] = (X_mat[:, 1:] - self.mean) / self.std
        # Solve normal equation with ridge regularization (L2)
        l2_reg = 0.1 * np.eye(X_mat.shape[1])
        l2_reg[0, 0] = 0.0 # No regularization for bias
        self.weights = np.linalg.pinv(X_mat.T @ X_mat + l2_reg) @ X_mat.T @ y
        
    def predict(self, X):
        X_mat = self._preprocess(X)
        X_mat = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])
        X_mat[:, 1:] = (X_mat[:, 1:] - self.mean) / self.std
        return X_mat @ self.weights

def compute_rmsle(y_true, y_pred):
    """
    Computes Root Mean Squared Logarithmic Error (RMSLE).
    Assumes inputs are already in the log-transformed space: log(sales + 1).
    """
    # Clip negative predictions to 0
    y_pred_clipped = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((y_true - y_pred_clipped) ** 2))

def run_pipeline(train_path, test_path, stores_path, oil_path, holidays_path, output_path):
    print("--- Use Case 1: Grocery Sales Forecasting Pipeline ---")
    
    # Import feature and validation modules locally
    from features import build_features
    from validation import get_time_series_split
    
    print("Step 1: Building Features...")
    train_feat, test_feat = build_features(
        train_path, test_path, stores_path, oil_path, holidays_path
    )
    
    print(f"Features created successfully. Total training samples: {len(train_feat)}")
    
    # Define features to use (exclude target and helper columns)
    exclude_cols = ['id', 'date', 'sales', 'log_sales']
    feature_cols = [c for c in train_feat.columns if c not in exclude_cols]
    
    # Print list of features being used
    print("Features selected for training:")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1}. {col} ({train_feat[col].dtype})")
        
    print("\nStep 2: Splitting Data Chronologically...")
    train_split, val_split = get_time_series_split(train_feat)
    
    X_train = train_split[feature_cols]
    y_train = train_split['log_sales']
    X_val = val_split[feature_cols]
    y_val = val_split['log_sales']
    
    print("\nStep 3: Training Model...")
    use_lightgbm = False
    try:
        import lightgbm as lgb
        use_lightgbm = True
    except ImportError:
        print("WARNING: LightGBM is not installed. Falling back to Ridge Baseline Regressor.")
        
    if use_lightgbm:
        # Define categorical features for LightGBM
        cat_features = [c for c in feature_cols if X_train[c].dtype.name == 'category']
        
        # Create dataset objects
        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=cat_features)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        # Train with early stopping
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        )
        
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds = model.predict(test_feat[feature_cols], num_iteration=model.best_iteration)
        print("LightGBM Model trained successfully.")
    else:
        # Run local fallback regression
        model = FallbackLinearRegression()
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(test_feat[feature_cols])
        print("Ridge Baseline Regressor trained successfully.")
        
    # Calculate Validation RMSLE
    val_rmsle = compute_rmsle(y_val.values, val_preds)
    print(f"\nValidation RMSLE: {val_rmsle:.5f}")
    
    print("\nStep 4: Error Analysis...")
    val_df_analysis = val_split.copy()
    val_df_analysis['pred_log_sales'] = val_preds
    val_df_analysis['pred_sales'] = np.expm1(val_preds)
    val_df_analysis['absolute_log_error'] = np.abs(val_df_analysis['log_sales'] - val_df_analysis['pred_log_sales'])
    
    # 1. Error by Product Family
    print("\nAverage Prediction Error (RMSLE) by Product Family (Top 10 highest error):")
    family_errors = val_df_analysis.groupby('family', observed=False).apply(
        lambda x: compute_rmsle(x['log_sales'].values, x['pred_log_sales'].values)
    ).sort_values(ascending=False)
    for family, err in family_errors.head(10).items():
         print(f"  {family:<30}: {err:.5f}")
         
    # 2. Error by Store Number
    print("\nAverage Prediction Error (RMSLE) by Store (Top 5 highest error):")
    store_errors = val_df_analysis.groupby('store_nbr').apply(
        lambda x: compute_rmsle(x['log_sales'].values, x['pred_log_sales'].values)
    ).sort_values(ascending=False)
    for store_nbr, err in store_errors.head(5).items():
         print(f"  Store {store_nbr:<27}: {err:.5f}")
         
    # 3. Impact of Holidays
    print("\nAverage Prediction Error (RMSLE) on Holidays vs. Normal Days:")
    holiday_errors = val_df_analysis.groupby('is_holiday').apply(
        lambda x: compute_rmsle(x['log_sales'].values, x['pred_log_sales'].values)
    )
    for is_h, err in holiday_errors.items():
         status = "Holiday" if is_h == 1 else "Normal Day"
         print(f"  {status:<30}: {err:.5f}")
         
    print("\nStep 5: Writing Kaggle Submission...")
    submission = pd.DataFrame({
        'id': test_feat['id'].astype(int),
        'sales': np.expm1(test_preds)
    })
    # Sales predictions cannot be negative
    submission['sales'] = submission['sales'].clip(0, None)
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Submission preview:\n{submission.head()}")
    print("--- Pipeline Completed Successfully ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time Series Forecasting for Corporacion Favorita")
    parser.add_argument('--train_path', type=str, required=True, help="Path to train.csv")
    parser.add_argument('--test_path', type=str, required=True, help="Path to test.csv")
    parser.add_argument('--stores_path', type=str, required=True, help="Path to stores.csv")
    parser.add_argument('--oil_path', type=str, required=True, help="Path to oil.csv")
    parser.add_argument('--holidays_path', type=str, required=True, help="Path to holidays_events.csv")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save predictions")
    
    args = parser.parse_args()
    
    run_pipeline(
        args.train_path,
        args.test_path,
        args.stores_path,
        args.oil_path,
        args.holidays_path,
        args.output_path
    )
