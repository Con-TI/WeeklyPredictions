import numpy as np
from itertools import permutations
import yfinance as yf
import pandas as pd

perms = list(permutations([0,1,2,3]))
perms = [np.array(perm)/3 for perm in perms]

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

def row_filter(row):
    array = np.array([row.iloc[0],row.iloc[1],row.iloc[2],row.iloc[3]])
    array = (array-array.min())/(array.max()-array.min())
    return_array = np.array([(np.abs(array-perm)).mean() for perm in perms])
    return pd.Series(return_array, index=[f'perm{i}' for i in range(len(perms))])

df[[f'perm{i}' for i in range(len(perms))]] = df.apply(func=row_filter,axis=1)
df['3_step_future_change'] = (close_prices.shift(-3) - close_prices)/close_prices*100
df = df.dropna()
df = df.drop(columns=columns)
df.to_pickle('contordpatt/test.pkl')