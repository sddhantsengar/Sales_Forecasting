import numpy as np

def ensemble_forecasts(*forecasts, weights=None):
    arr = np.array(forecasts)
    if weights is None:
        weights = np.ones(len(forecasts)) / len(forecasts)
    return np.average(arr, axis=0, weights=weights)
