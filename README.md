# Multi-Store Sales Forecasting

Forecasts daily sales per store-item combination using three time series
approaches — **ARIMA**, **Facebook Prophet**, and **LSTM** — and combines
them with a simple ensemble.

## Project Structure

```
multi-store-sales-forecasting/
├── data/
│   └── generate_data.py     # creates a synthetic sample dataset
├── src/
│   ├── data_utils.py        # load / split time series
│   ├── arima_model.py       # ARIMA fit + forecast
│   ├── prophet_model.py     # Prophet fit + forecast
│   ├── lstm_model.py        # LSTM fit + forecast
│   ├── ensemble.py          # weighted average of forecasts
│   └── evaluate.py          # RMSE / MAE metrics
├── main.py                  # end-to-end pipeline for one store-item pair
├── requirements.txt
└── outputs/                 # forecast plot saved here
```

## Setup

```bash
git clone <your-repo-url>
cd multi-store-sales-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data

This repo ships with a small synthetic dataset generator so it runs
out of the box:

```bash
python data/generate_data.py
```

To use the real dataset instead, download the
[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)
data from Kaggle and save it as `data/sample_data.csv` with columns
`date, store, item, sales`.

## Run

```bash
python main.py
```

This fits ARIMA, Prophet, and an LSTM on one store-item series, forecasts
the last 90 days, ensembles the three forecasts, prints RMSE/MAE for each,
and saves a comparison plot to `outputs/forecast.png`.

## Method

| Model   | What it captures                                  |
|---------|-----------------------------------------------------|
| ARIMA   | Linear trend + autocorrelation in a single series   |
| Prophet | Trend + weekly/yearly seasonality, holiday effects  |
| LSTM    | Nonlinear temporal patterns learned from the data   |
| Ensemble| Weighted average of the three forecasts             |

## License

MIT
"# Sales_Forecasting" 
"# Sales_Forecasting" 
