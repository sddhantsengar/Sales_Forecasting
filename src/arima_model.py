from statsmodels.tsa.arima.model import ARIMA

def fit_arima(train, order=(5, 1, 0)):
    return ARIMA(train, order=order).fit()

def forecast_arima(model, steps):
    return model.forecast(steps=steps)
