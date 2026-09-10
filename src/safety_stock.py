"""SKU-level safety-stock / reorder-point table.

Method: for each store-item, demand over a 7-day lead time is assumed normal;
reorder point = mean demand over lead time + z(0.95) * std_daily * sqrt(L).
Outputs the top-20 SKUs by average daily demand.
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
LEAD_TIME_DAYS = 7
Z = 1.645  # 95% service level


def main():
    df = pd.read_csv(DATA, parse_dates=['date'])
    g = df.groupby(['store', 'item'])['sales']
    stats_df = pd.DataFrame({
        'mean_daily': g.mean(),
        'std_daily': g.std(),
        'total_sales': g.sum(),
    }).reset_index()

    mu_L = stats_df['mean_daily'] * LEAD_TIME_DAYS
    sigma_L = stats_df['std_daily'] * np.sqrt(LEAD_TIME_DAYS)
    stats_df['safety_stock'] = np.ceil(Z * sigma_L).astype(int)
    stats_df['reorder_point'] = np.ceil(mu_L + Z * sigma_L).astype(int)

    top = (stats_df.sort_values('mean_daily', ascending=False)
           .head(20)[['store', 'item', 'mean_daily', 'std_daily',
                      'safety_stock', 'reorder_point']])
    top['mean_daily'] = top['mean_daily'].round(1)
    top['std_daily'] = top['std_daily'].round(1)
    os.makedirs(OUT, exist_ok=True)
    top.to_csv(os.path.join(OUT, 'safety_stock_top20.csv'), index=False)
    print(top.to_string(index=False))
    print(f"\nassumptions: lead time L={LEAD_TIME_DAYS}d, service level 95% (z={Z})")


if __name__ == '__main__':
    main()
