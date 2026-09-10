"""High-demand (promotion/holiday) period uplift analysis.

For each item (all stores pooled): days in the item's top decile of demand are
treated as high-demand (promotional/holiday) periods; uplift is measured
against the item's median-day baseline.
"""
import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    df = pd.read_csv(DATA, parse_dates=['date'])
    item = df.groupby(['item', 'date'])['sales'].sum().reset_index()

    item['baseline'] = item.groupby('item')['sales'].transform('median')
    item['rank'] = item.groupby('item')['sales'].rank(pct=True)
    item['high'] = item['rank'] >= 0.9

    agg = (item.groupby('item')
           .apply(lambda g: pd.Series({
               'baseline_daily': g['baseline'].iloc[0],
               'high_period_daily': g.loc[g['high'], 'sales'].mean(),
               'uplift_pct': (g.loc[g['high'], 'sales'].mean() / g['baseline'].iloc[0] - 1) * 100,
           }), include_groups=False)
           .reset_index())

    summary = pd.DataFrame([{
        'items': len(agg),
        'median_uplift_pct': round(agg['uplift_pct'].median(), 1),
        'p25_uplift_pct': round(agg['uplift_pct'].quantile(0.25), 1),
        'p75_uplift_pct': round(agg['uplift_pct'].quantile(0.75), 1),
        'top_item': int(agg.loc[agg['uplift_pct'].idxmax(), 'item']),
        'top_item_uplift_pct': round(agg['uplift_pct'].max(), 1),
    }])
    os.makedirs(OUT, exist_ok=True)
    agg.round(1).to_csv(os.path.join(OUT, 'high_demand_uplift_by_item.csv'), index=False)
    summary.to_csv(os.path.join(OUT, 'high_demand_uplift_summary.csv'), index=False)
    print(summary.to_string(index=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(agg['uplift_pct'], bins=30, color='#4a72a8', edgecolor='white')
    ax.axvline(agg['uplift_pct'].median(), color='#c0504d', ls='--',
               label=f"median +{agg['uplift_pct'].median():.0f}%")
    ax.set_title('High-demand period sales uplift vs median-day baseline, by item')
    ax.set_xlabel('uplift %')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'high_demand_uplift.png'), dpi=150)
    print('saved results/high_demand_uplift_*.csv, high_demand_uplift.png')


if __name__ == '__main__':
    main()
