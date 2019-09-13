import csv
import numpy as np
from sklearn import linear_model #regression line
import matplotlib.pyplot as plt
from datetime import datetime

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

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

# def show_plot(dates, prices):
#     linear_mod = linear_model.LinearRegression()
#     dates = np.reshape(dates, (len(dates), 1))



get_data("AAPL.csv")

print(dates[0])
print(prices[0])

# plt.plot([1,2,5,4])
# plt.show()
print(prices)
plt.plot(dates, prices)
plt.show()


# data = range(100)
# dataRange = range(len(data))
# plt.scatter(dataRange, data)
# plt.show()