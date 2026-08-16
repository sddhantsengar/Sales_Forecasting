import pandas as pd

def load_data(path):
    return pd.read_csv(path, parse_dates=['date'])

def get_series(df, store, item):
    s = df[(df.store == store) & (df.item == item)].sort_values('date')
    return s.set_index('date')['sales']

def train_test_split(series, test_days=90):
    return series.iloc[:-test_days], series.iloc[-test_days:]
