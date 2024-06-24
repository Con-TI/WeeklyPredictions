import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num = 15

df = yf.download(tickers=['BTC-USD'], period='max')
close_prices = df['Close']
diffs = []
diff = close_prices.pct_change()
diffs.append(diff)
for i in range(1,num):
    diffs.append(diff.shift(i))
df = pd.concat(diffs,axis=1)
df.columns = [f'shift_{i}' for i in range(num)]

def min_max_rows(row):
    array = np.array([row.iloc[i] for i in range(num)])
    array = ((array-array.min())/(array.max()-array.min()))*2-1
    array = np.arccos(array)
    return pd.Series(array,index=[f'shift_{i}' for i in range(num)])

arrays = pd.DataFrame(df.apply(func=min_max_rows,axis=1),columns=[f'shift_{i}' for i in range(num)]).dropna()
    
def sum_cos_gram_matrix(row):
    array = np.array([row.iloc[i] for i in range(num)])
    matrix = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            matrix[i,j] = np.cos(array[i] + array[j])
    return matrix

def diff_cos_gram_matrix(row):
    array = np.array([row.iloc[i] for i in range(num)])
    matrix = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            matrix[i,j] = np.cos(array[i] - array[j])
    return matrix

def zeros(row):
    return np.zeros((num,num))

gram = pd.concat([arrays.apply(func=sum_cos_gram_matrix,axis=1),arrays.apply(func=diff_cos_gram_matrix,axis=1), arrays.apply(func=zeros,axis=1)],axis=1)

def combine(row):
    matrix = np.stack([row.iloc[i] for i in range(3)])
    return (matrix-matrix.min())/(matrix.max()-matrix.min())

gram = gram.apply(func=combine,axis=1)
correct_preds = 100*(close_prices.shift(-7) - close_prices)/close_prices
df = pd.concat([gram,correct_preds], axis=1).dropna()
df.columns = ['matrix', 'y']
df.to_pickle('GramianAngularField/test.pkl')

# fig, ax = plt.subplots()
# img = ax.imshow(np.zeros((num,num,3)))

# def update(frame):
#     matrix = gram.iloc[frame]
#     img.set_data(matrix.transpose(1,2,0))
#     ax.set_title(f'Frame {frame}')
#     return img,

# anim = FuncAnimation(fig, update, frames= len(gram), interval = 10, blit = True)

# plt.colorbar(img)
# plt.show()