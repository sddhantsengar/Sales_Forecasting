import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

def fit_lstm(train, window=30, epochs=20):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    X, y = create_sequences(scaled, window)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(window, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler

def forecast_lstm(model, scaler, train, steps, window=30):
    scaled = scaler.transform(train.values.reshape(-1, 1))
    seq = list(scaled[-window:].flatten())
    preds = []
    for _ in range(steps):
        x = np.array(seq[-window:]).reshape((1, window, 1))
        pred = model.predict(x, verbose=0)[0][0]
        preds.append(pred)
        seq.append(pred)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds.flatten()
