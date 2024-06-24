import numpy as np
from itertools import permutations
import yfinance as yf
import pandas as pd

df = yf.download(tickers=['BTC-USD'], period='max')
shifted = []
close_prices = df['Close']

diff = close_prices.diff(1)
gain, loss = diff, -diff
gain[diff<0] = 0
loss[diff>0] = 0
n = 14
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain/avg_loss
rsi = 100 - (100/(1+rs))

shifted.append(rsi)
for i in range(1,4):
    shifted.append(rsi.shift(i))
df = pd.concat(shifted,axis=1)
columns = [f"Shift_{i}" for i in range(4)]
df.columns = columns

def process_row(row):
    values = [row.iloc[0],row.iloc[1],row.iloc[2],row.iloc[3]]
    sorted_values = sorted((val, idx) for idx,val in enumerate(values))
    ranks = [0]*len(values)
    for rank, (val,idx) in enumerate(sorted_values,start=1):
        ranks[idx] = rank
    return_array = np.array(ranks)
    return pd.Series(return_array, index=[f'val{i}' for i in range(4)])

df[[f'val{i}' for i in range(4)]] = df.apply(func=process_row,axis=1)
df = df.drop(columns=columns)
df = pd.concat([df,df.shift(3),df.shift(6)],axis=1)
df['3_step_future_change'] = (close_prices.shift(-3) - close_prices)/close_prices*100
df = df.dropna()
df.to_pickle('discordpatt/test.pkl')