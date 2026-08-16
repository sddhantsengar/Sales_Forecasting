import os
import matplotlib.pyplot as plt
from src.data_utils import load_data, get_series, train_test_split
from src.arima_model import fit_arima, forecast_arima
from src.prophet_model import fit_prophet, forecast_prophet
from src.lstm_model import fit_lstm, forecast_lstm
from src.ensemble import ensemble_forecasts
from src.evaluate import rmse, mae

STORE, ITEM, TEST_DAYS = 1, 1, 90

def main():
    df = load_data('data/sample_data.csv')
    series = get_series(df, STORE, ITEM)
    train, test = train_test_split(series, TEST_DAYS)

    arima_model = fit_arima(train)
    arima_fc = forecast_arima(arima_model, TEST_DAYS)

    prophet_model = fit_prophet(train)
    prophet_fc = forecast_prophet(prophet_model, TEST_DAYS)

    lstm_model, scaler = fit_lstm(train)
    lstm_fc = forecast_lstm(lstm_model, scaler, train, TEST_DAYS)

    ensemble_fc = ensemble_forecasts(arima_fc.values, prophet_fc.values, lstm_fc)

    results = {
        'ARIMA': (rmse(test, arima_fc), mae(test, arima_fc)),
        'Prophet': (rmse(test, prophet_fc), mae(test, prophet_fc)),
        'LSTM': (rmse(test, lstm_fc), mae(test, lstm_fc)),
        'Ensemble': (rmse(test, ensemble_fc), mae(test, ensemble_fc)),
    }

    for name, (r, m) in results.items():
        print(f'{name}: RMSE={r:.2f}  MAE={m:.2f}')

    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(test.index, arima_fc.values, label='ARIMA')
    plt.plot(test.index, prophet_fc.values, label='Prophet')
    plt.plot(test.index, lstm_fc, label='LSTM')
    plt.plot(test.index, ensemble_fc, label='Ensemble')
    plt.legend()
    plt.title(f'Store {STORE} Item {ITEM} Sales Forecast')
    plt.savefig('outputs/forecast.png')

if __name__ == '__main__':
    main()
