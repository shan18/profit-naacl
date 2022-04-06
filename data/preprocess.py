import os
import argparse
import pickle
from tqdm import tqdm
import torch
import json
import pandas as pd
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertModel


MAX_TWEETS = 30
N_DAYS = 7
TWEETS_EMB = 768
START_DATE = datetime(year=2015, day=1, month=10)
END_DATE = datetime(year=2016, day=1, month=1)
DELTA_DAYS = timedelta(days=N_DAYS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased').to(device)


def get_date_sliced_df(dfp, dft, start_date, end_date):
    return (
        dft[(dft >= start_date) & (dft < end_date)],
        dfp[(dfp.date >= start_date) & (dfp.date < end_date)]
    )


def get_closing_prices(dfp_w, data):
    adj_close_last = dfp_w.open_price.iloc[0]
    if len(data) > 0 and 'adj_close_target' in data[-1]:
        adj_close_last = data[-1]['adj_close_target']
    adj_close_target = dfp_w.close_price.iloc[-1]
    return torch.tensor([adj_close_last]), torch.tensor([adj_close_target])


def read_tweet_json(path):
    json_data = []
    inv_time = []
    prev_time = None
    with open(path) as f:
        for line in f:
            tweet_info = json.loads(line)
            tweet_time = datetime.strptime(tweet_info['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            prev_time = tweet_time if prev_time is None else prev_time

            json_data.append(' '.join(tweet_info['text']))
            inv_time.append(0 if tweet_time == prev_time else 1 / (tweet_time - prev_time).seconds)

            prev_time = tweet_time

    return (json_data, inv_time) if len(json_data) <= MAX_TWEETS else (json_data[:MAX_TWEETS], inv_time[:MAX_TWEETS])


def get_day_tweets(dft_w, start_date, end_date, company_tweet_path):
    day_tweets = []
    day_tweets_len = []
    inv_time = []
    while start_date < end_date:
        tday_str = start_date.strftime('%Y-%m-%d')
        if dft_w[start_date == dft_w].empty:
            day_tweets.append(['' for _ in range(MAX_TWEETS)])
            day_tweets_len.append(0)
            inv_time.append([0 for _ in range(MAX_TWEETS)])
        else:
            day_tweets_tday, inv_day_time = read_tweet_json(os.path.join(company_tweet_path, tday_str))
            day_tweets_len.append(len(day_tweets_tday))
            if day_tweets_len[-1] < MAX_TWEETS:
                day_tweets_tday += ['' for _ in range(MAX_TWEETS - len(day_tweets_tday))]
                inv_day_time += [0 for _ in range(MAX_TWEETS - len(inv_day_time))]
            day_tweets.append(day_tweets_tday)
            inv_time.append(inv_day_time)

        start_date += timedelta(days=1)

    return (
        day_tweets, torch.tensor(day_tweets_len).unsqueeze(0),
        torch.tensor(inv_time).unsqueeze(-1).unsqueeze(0)
    )


def get_bert_embeddings(tweet_w):
    tweet_w_extended = []
    for i in tweet_w:
        tweet_w_extended.extend(i)

    tokens = tokenizer(tweet_w_extended, padding=True, return_tensors='pt')['input_ids']
    return model(tokens.to(device))[0][:, 0, :].view(N_DAYS, MAX_TWEETS, -1).detach().cpu().unsqueeze(0)


def process_data(tweet_path, price_path, output_path):
    print('Processing...')
    pbar = tqdm(os.listdir(tweet_path))
    for company in pbar:
        pbar.set_description(f'Processing {company}')
        output_file = os.path.join(output_path, f'{company}.pkl')
        if os.path.exists(output_file):
            continue

        dfp = pd.read_csv(os.path.join(price_path, f'{company}.csv'))
        dfp.date = pd.to_datetime(dfp.date)
        dft = pd.Series(sorted([datetime.strptime(x, '%Y-%m-%d') for x in os.listdir(os.path.join(tweet_path, company))]))

        data = []
        current_date = START_DATE
        while current_date <= END_DATE:
            window = {}
            current_last_date = current_date + DELTA_DAYS
            dft_w, dfp_w = get_date_sliced_df(dfp, dft, current_date, current_last_date)
            if not dft_w.empty and not dfp_w.empty:
                window['adj_close_last'], window['adj_close_target'] = get_closing_prices(dfp_w, data)
                tweet_w, window['length_data'], window['time_features'] = get_day_tweets(
                    dft_w, current_date, current_last_date, os.path.join(tweet_path, company)
                )
                window['embedding'] = get_bert_embeddings(tweet_w)
            else:
                window['adj_close_last'], window['adj_close_target'] = torch.tensor([0]), torch.tensor([0])
                window['length_data'] = torch.zeros((1, N_DAYS))
                window['embedding'] = torch.zeros((1, N_DAYS, MAX_TWEETS, TWEETS_EMB))
                window['time_features'] = torch.zeros((1, N_DAYS, MAX_TWEETS, 1))
            data.append(window)
            current_date += DELTA_DAYS

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
    print('Done.')


def merge_data(data_path, merge_path):
    print('\nMerging...')
    merged_data = []
    pbar = tqdm(os.listdir(data_path))
    for company in pbar:
        pbar.set_description(f'Merging {company}')
        with open(os.path.join(data_path, company), 'rb') as f:
            data = pickle.load(f)

        if merged_data == []:
            merged_data = data
        else:
            for i in range(len(data)):
                merged_data[i]['adj_close_last'] = torch.cat((merged_data[i]['adj_close_last'], data[i]['adj_close_last']), dim=0)
                merged_data[i]['adj_close_target'] = torch.cat((merged_data[i]['adj_close_target'], data[i]['adj_close_target']), dim=0)
                merged_data[i]['length_data'] = torch.cat((merged_data[i]['length_data'], data[i]['length_data']), dim=0)
                merged_data[i]['embedding'] = torch.cat((merged_data[i]['embedding'], data[i]['embedding']), dim=0)
                merged_data[i]['time_features'] = torch.cat((merged_data[i]['time_features'], data[i]['time_features']), dim=0)
    print('Done.')

    print('\nSaving...')
    with open(merge_path, 'wb') as f:
        pickle.dump(merged_data, f)
    print('Done.')


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweet', default=os.path.join(BASE_DIR, 'tweet'), help='Path to directory containing tweets')
    parser.add_argument('--price', default=os.path.join(BASE_DIR, 'price'), help='Path to directory containing stock prices')
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'stock'), help='Path to directory containing processed stock prices')
    parser.add_argument('--merge',default=os.path.join(BASE_DIR, 'stock_data.pkl'), help='Merge data file')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    process_data(args.tweet, args.price, args.output)
    merge_data(args.output, args.merge)
