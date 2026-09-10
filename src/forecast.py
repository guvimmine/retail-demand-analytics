"""Demand forecasting baselines + gradient-boosting model on store-item daily sales.

Data: Kaggle "Store Item Demand Forecasting" (10 stores x 50 items, 2013-2017).
Validation: last 28 days. Metrics: MAE and WAPE.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
VAL_DAYS = 28


def load():
    df = pd.read_csv(DATA, parse_dates=['date'])
    return df.sort_values(['store', 'item', 'date']).reset_index(drop=True)


def seasonal_naive(train, valid):
    """Weekly replay: forecast for date d = demand at d-7 (replayed back into train)."""
    last = train[train['date'] > train['date'].max() - pd.Timedelta(days=7)]
    lut = {(s, i, d): v for s, i, d, v in
           zip(last['store'], last['item'], last['date'], last['sales'])}
    out = valid.copy()
    preds = []
    for s, i, d in zip(valid['store'], valid['item'], valid['date']):
        dd = d.to_timestamp() if hasattr(d, 'to_timestamp') else d
        while (s, i, dd) not in lut:
            dd -= pd.Timedelta(days=7)
        preds.append(lut[(s, i, dd)])
    out['pred'] = preds
    return out


def add_features(df):
    df = df.copy()
    for lag in (7, 14, 28):
        df[f'lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df


def main():
    df = load()
    cut = df['date'].max() - pd.Timedelta(days=VAL_DAYS)
    train, valid = df[df['date'] <= cut], df[df['date'] > cut]

    # --- baseline: seasonal naive -----------------------------------------
    snaive = seasonal_naive(train, valid)

    # --- gradient boosting -------------------------------------------------
    feats = add_features(df)
    f_train = feats[feats['date'] <= cut].dropna(subset=['lag_7'])
    f_valid = feats[feats['date'] > cut].dropna(subset=['lag_7'])
    cols = ['lag_7', 'lag_14', 'lag_28', 'dow', 'month', 'day', 'store', 'item']
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42)
    model.fit(f_train[cols], f_train['sales'])
    f_valid = f_valid.copy()
    f_valid['pred'] = np.maximum(model.predict(f_valid[cols]), 0)

    # --- metrics ------------------------------------------------------------
    def metrics(pred_df, name):
        y = pred_df['sales'].to_numpy()
        p = pred_df['pred'].to_numpy()
        mae = np.abs(y - p).mean()
        wape = np.abs(y - p).sum() / y.sum() * 100
        return {'model': name, 'MAE': round(mae, 2), 'WAPE_%': round(wape, 2)}

    rows = [metrics(snaive, 'seasonal_naive_lag7'),
            metrics(f_valid, 'hist_gradient_boosting')]

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'forecast_metrics.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # sample chart: store 1 item 1, last 90 days
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = df[(df['store'] == 1) & (df['item'] == 1)].tail(90)
    p = f_valid[(f_valid['store'] == 1) & (f_valid['item'] == 1)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s['date'], s['sales'], label='actual')
    ax.plot(p['date'], p['pred'], label='gbm forecast')
    ax.set_title('Store 1 / Item 1 — last 90 days')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'forecast_sample.png'), dpi=150)
    print('saved results/forecast_metrics.csv, forecast_sample.png')


if __name__ == '__main__':
    main()
