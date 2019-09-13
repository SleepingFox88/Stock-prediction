import csv
import numpy as np
from sklearn import linear_model #regression line
import matplotlib.pyplot as plt
from datetime import datetime

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


get_data("AAPL.csv")

# plot all our price data
# plt.plot(dates, prices)
# plt.show()

# number of samles with at least 32 entried of data before it.
# we need past 32 days to learn and make predicitons
window_size=32
num_samples=len(dates)-window_size

# get X data
x_data = []
for x in range(num_samples - 1):
    past32Prices = []
    for x2 in range(x,x+32):
        past32Prices.append(prices[x])
    x_data.append(past32Prices)

# get y data
y_data = []
for x in range(num_samples - 1):
    y_data.append(prices[x+33])


# we want to use 80% of our data for training, and 20% for testing
# This finds the array index where we should split our data
split_fraction = 0.8
ind_split=int(split_fraction * num_samples)

# set trainging and test data
x_train = x_data[:ind_split]
y_train = y_data[:ind_split]
x_test = x_data[ind_split:]
y_test = y_data[ind_split:]