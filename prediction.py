import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 1, 11)

df = web.DataReader("AAPL", 'yahoo', start, end)
df.tail()


close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()


import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the style of matplotlib
style.use('ggplot')

mavg.plot(label='mavg')
close_px.plot(label='close_px')
plt.ylabel('Value')
plt.xlabel('Year')
plt.show()