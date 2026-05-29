import os
import pandas as pd
import numpy as np

def generate_synthetic_data(data_dir):
    """
    Generates a small synthetic dataset for testing the forecasting pipeline.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    np.random.seed(42)
    start_date = '2017-06-01'
    end_date = '2017-08-31'
    dates = pd.date_range(start=start_date, end=end_date)
    
    store_nbrs = [1, 2]
    families = ['GROCERY I', 'BEVERAGES']
    
    # 1. Create stores.csv
    stores = pd.DataFrame({
        'store_nbr': store_nbrs,
        'city': ['Quito', 'Guayaquil'],
        'state': ['Pichincha', 'Guayas'],
        'type': ['D', 'B'],
        'cluster': [13, 6]
    })
    stores.to_csv(os.path.join(data_dir, 'stores.csv'), index=False)
    
    # 2. Create oil.csv (with missing dates to test interpolation)
    oil = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'dcoilwtico': [45.0 + np.sin(i/10) + np.random.normal(0, 0.5) for i in range(len(dates))]
    })
    # Remove some entries (simulate weekends/holidays)
    oil.loc[oil.index % 7 >= 5, 'dcoilwtico'] = np.nan
    oil.to_csv(os.path.join(data_dir, 'oil.csv'), index=False)
    
    # 3. Create holidays_events.csv
    holidays = pd.DataFrame([
        {'date': '2017-06-25', 'type': 'Holiday', 'locale': 'Local', 'locale_name': 'Quito', 'description': 'Fundacion de Quito', 'transferred': False},
        {'date': '2017-07-24', 'type': 'Transfer', 'locale': 'Local', 'locale_name': 'Guayaquil', 'description': 'Fundacion de Guayaquil', 'transferred': False},
        {'date': '2017-08-10', 'type': 'Holiday', 'locale': 'National', 'locale_name': 'Ecuador', 'description': 'Primer Grito de Independencia', 'transferred': False}
    ])
    holidays.to_csv(os.path.join(data_dir, 'holidays_events.csv'), index=False)
    
    # 4. Create train.csv (dates from June 1st to August 15th)
    train_dates = pd.date_range(start=start_date, end='2017-08-15')
    train_records = []
    record_id = 0
    for dt in train_dates:
        for store in store_nbrs:
            for fam in families:
                base_sales = 100 + 50 * np.sin(dt.dayofweek / 7 * 2 * np.pi) + np.random.exponential(10)
                train_records.append({
                    'id': record_id,
                    'date': dt.strftime('%Y-%m-%d'),
                    'store_nbr': store,
                    'family': fam,
                    'sales': max(0.0, base_sales),
                    'onpromotion': int(np.random.rand() > 0.8)
                })
                record_id += 1
                
    train = pd.DataFrame(train_records)
    train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    
    # 5. Create test.csv (dates from August 16th to August 31st)
    test_dates = pd.date_range(start='2017-08-16', end=end_date)
    test_records = []
    for dt in test_dates:
        for store in store_nbrs:
            for fam in families:
                test_records.append({
                    'id': record_id,
                    'date': dt.strftime('%Y-%m-%d'),
                    'store_nbr': store,
                    'family': fam,
                    'onpromotion': int(np.random.rand() > 0.8)
                })
                record_id += 1
                
    test = pd.DataFrame(test_records)
    test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    print("Synthetic data generated successfully.")

if __name__ == '__main__':
    data_dir = 'grocery_forecasting/data'
    generate_synthetic_data(data_dir)
    
    # Execute training command
    import subprocess
    cmd = [
        'python', 'grocery_forecasting/src/model.py',
        '--train_path', 'grocery_forecasting/data/train.csv',
        '--test_path', 'grocery_forecasting/data/test.csv',
        '--stores_path', 'grocery_forecasting/data/stores.csv',
        '--oil_path', 'grocery_forecasting/data/oil.csv',
        '--holidays_path', 'grocery_forecasting/data/holidays_events.csv',
        '--output_path', 'grocery_forecasting/data/submission.csv'
    ]
    
    print("\nExecuting forecasting pipeline...")
    # Add src to python path
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath('grocery_forecasting/src')
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    if res.returncode != 0:
        print("STDERR:")
        print(res.stderr)
        raise RuntimeError("Forecasting pipeline test failed.")
