# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 100000
# total number of stocks in our portfolio
STOCK_DIM = 87
N_DAYS = 7
MAX_TWEETS = 30
TWEETS_EMB = 768
TIME_FEATS = STOCK_DIM * N_DAYS * MAX_TWEETS
# Shape:  [Current Balance] + [Last Day Prices] + [Target Day Prices] + [Owned Shares] + [Text Features + Tweet Length]
FEAT_DIMS = (
    1
    + STOCK_DIM * 2
    + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB
    + STOCK_DIM * N_DAYS
    + STOCK_DIM
    + TIME_FEATS
)

# all starting indexes
# [INITIAL_ACCOUNT_BALANCE]  # balance
# + last_price  # stock prices initial
# + [0] * STOCK_DIM  # stocks on hold
# + emb_data  # tweet features
# + len_data  # tweet len
# + target_price  # target price
# + time_feats

LAST_PRICE_IDX = 1
HOLDING_IDX = LAST_PRICE_IDX + STOCK_DIM
EMB_IDX = HOLDING_IDX + STOCK_DIM
LEN_IDX = EMB_IDX + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB
TARGET_IDX = LEN_IDX + STOCK_DIM * N_DAYS
TIME_IDX = TARGET_IDX + STOCK_DIM
TRANSACTION_FEE_PERCENT = 0.1
REWARD_SCALING = 1e-4