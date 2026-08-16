import numpy as np
import pandas as pd

def generate(stores=3, items=5, start='2018-01-01', end='2022-12-31', seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq='D')
    rows = []
    for store in range(1, stores + 1):
        for item in range(1, items + 1):
            base = rng.uniform(20, 60)
            trend = np.linspace(0, rng.uniform(5, 20), len(dates))
            weekly = 8 * np.sin(2 * np.pi * dates.dayofweek / 7)
            yearly = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)
            noise = rng.normal(0, 4, len(dates))
            sales = np.clip(base + trend + weekly + yearly + noise, 0, None).round().astype(int)
            rows.append(pd.DataFrame({'date': dates, 'store': store, 'item': item, 'sales': sales}))
    return pd.concat(rows, ignore_index=True)

if __name__ == '__main__':
    df = generate()
    df.to_csv('data/sample_data.csv', index=False)
    print(f'Saved {len(df)} rows to data/sample_data.csv')
