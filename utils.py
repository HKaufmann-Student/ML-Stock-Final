import pandas as pd
import numpy as np

def predict_top_stocks(model, recent_data, scaler, label_encoder, features, window_size=60):
    scaled_data = scaler.transform(recent_data[features])
    seq = scaled_data[-window_size:].reshape(1, window_size, -1)
    prob = model.predict(seq).flatten()

    unique_ticker_enc = recent_data['Ticker_encoded'].unique()
    preds = []
    for enc in unique_ticker_enc:
        preds.append((enc, prob[0]))

    df_preds = pd.DataFrame(preds, columns=['Ticker_encoded', 'Prediction'])
    df_preds['Ticker'] = label_encoder.inverse_transform(df_preds['Ticker_encoded'])
    df_preds.sort_values('Prediction', ascending=False, inplace=True)
    return df_preds
