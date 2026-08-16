from prophet import Prophet

def fit_prophet(train):
    df = train.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    return model

def forecast_prophet(model, steps):
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat'].iloc[-steps:]
