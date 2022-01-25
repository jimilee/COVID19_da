import math
import numpy as np
import pandas as pd
def smape_loss(y_true, y_pred):
    return np.abs(y_true - y_pred) / (y_true + np.abs(y_pred)) * 200

def engineer(df):
    """Return a new dataframe with the engineered features"""
    new_df = pd.DataFrame({
        'wd4': df.date.dt.weekday == 4,  # Friday
        'wd56': df.date.dt.weekday >= 5,  # Saturday and Sunday
    })
    new_df['domestic'] = 1
    new_df['inflow'] = 1
    # Seasonal variations (Fourier series)
    # The three products have different seasonal patterns
    dayofyear = df.date.dt.dayofyear
    for k in range(1, 4):
        print(k, df.keys())
        new_df[f'sin{k}'] = np.sin(dayofyear / 365 * 2 * math.pi * k)
        new_df[f'cos{k}'] = np.cos(dayofyear / 365 * 2 * math.pi * k)
        new_df[f'total_sin{k}'] = new_df[f'sin{k}'] * df['total']
        new_df[f'total_cos{k}'] = new_df[f'cos{k}'] * df['total']
        new_df[f'domestic_sin{k}'] = new_df[f'sin{k}'] * new_df['domestic']
        new_df[f'domestic_cos{k}'] = new_df[f'cos{k}'] * new_df['domestic']
        new_df[f'inflow_sin{k}'] = new_df[f'sin{k}'] * new_df['inflow']
        new_df[f'inflow_cos{k}'] = new_df[f'cos{k}'] * new_df['inflow']

    return new_df