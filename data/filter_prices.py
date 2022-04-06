import argparse
import os
import pandas as pd
from datetime import datetime, timedelta


N_DAYS = 7


def filter_company_prices(dfp, dft):
    # Convert to datetime
    dfp.date = pd.to_datetime(dfp.date)
    dfp = dfp.sort_values(by='date')

    # Filter rows
    idx_list = []
    for i in range(len(dfp)):
        fil = dft[(dft > dfp.date.iloc[i] - timedelta(days=7)) & (dft < dfp.date.iloc[i])]
        if fil.empty:
            idx_list.append(i)

    return dfp.drop(idx_list)


def filter_stock_data(tweet_path, price_path, output_path):
    for company in os.listdir(tweet_path):
        dfp = pd.read_csv(os.path.join(price_path, company + '.csv'))
        dft = pd.Series(sorted([datetime.strptime(x, '%Y-%m-%d') for x in os.listdir(os.path.join(tweet_path, company))]))
        df_filtered = filter_company_prices(dfp, dft)
        df_filtered.to_csv(os.path.join(output_path, company + '.csv'), index=False)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweet', default=os.path.join(BASE_DIR, 'tweet'), help='Path to directory containing tweets')
    parser.add_argument('--price', default=os.path.join(BASE_DIR, 'price'), help='Path to directory containing stock prices')
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'price_new'), help='Path to directory containing filtered stock prices')
    args = parser.parse_args()

    filter_stock_data(args.tweet, args.price, args.output)
