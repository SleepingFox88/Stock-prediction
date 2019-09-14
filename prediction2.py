from datetime import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt

#Modeling
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model

#Modeling Metrics
from sklearn import metrics

# Loads price CSV data into dates and prices variables
dates = []
prices = []
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # get field headers on first line of csv file
        headerFields = next(csvreader)
        # load file data into rows[row][columb]
        rows = []
        for row in csvreader:
            rows.append(row)

        for x in range(len(rows)):

            date_string = rows[x][headerFields.index("Date")]
            format = "%Y-%m-%d"
            curDateTime = datetime.strptime(date_string, format)

            dates.append(curDateTime)
            prices.append(float(rows[x][headerFields.index("Adj Close")]))
    return

#Help Functions
def get_performance (model_pred):
  #Function returns standard performance metrics
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, model_pred).round(4))  
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, model_pred).round(4))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, model_pred)).round(4))
  
def get_plot (model_pred):
  plt.scatter(model_pred, y_test, color="gray")
  plt.plot(y_test, y_test, color='red', linewidth=2)
  plt.show()

get_data("AAPL.csv")

# number of samles with at least 32 entried of data before it.
# we need past 32 days to learn and make predicitons
window_size = 32
num_samples = len(prices) - window_size

# get X data
# each x_data item is an array of 32 days of price data, used to predict the 33rd day's price (y_data)
x_data = []
for x in range(num_samples):
    past32Prices = []
    for x2 in range(x, x + window_size):
        past32Prices.append(prices[x2])
    x_data.append(past32Prices)

# get y data
y_data = []
for x in range(num_samples):
    y_data.append(prices[x + window_size])


# we want to use 80% of our data for training, and 20% for testing
# This finds the array index where we should split our data
split_fraction = 0.8
ind_split=int(split_fraction * num_samples)

# set trainging and test data
x_train = x_data[:ind_split]
y_train = y_data[:ind_split]
x_test = x_data[ind_split:]
y_test = y_data[ind_split:]

# it apprears my x dataset is messed up
print(x_train[0])

print("x_test[0]")
print(x_test[0])

print("ytest[0]")
print(y_test[0])

# base tests
print("\nBase tests")
y_pred_lag=np.roll(y_test,1)
get_performance(y_pred_lag)
get_plot(y_pred_lag)

# linear regression
model_lr = LinearRegression()
# train on this data
model_lr.fit(x_train, y_train)
#generate predictions
y_pred_lr = model_lr.predict(x_test)
# results
print("\nLinear regression")
get_performance(y_pred_lr)
get_plot(y_pred_lr)

# Ridge Regression¶
model_ridge = Ridge()
# train on this data
model_ridge.fit(x_train, y_train)
#generate predictions
y_pred_ridge=model_ridge.predict(x_test)
# results
print("\nRidge")
get_performance(y_pred_ridge)
get_plot(y_pred_ridge)


# Gradient Boosting Trees
model_gb = GradientBoostingRegressor()
# train on this data
model_gb.fit(x_train, y_train)
# Infer
y_pred_gb = model_gb.predict(x_test)
# results
print("\nGradient Boosting Trees")
get_performance(y_pred_gb)
get_plot(y_pred_gb)