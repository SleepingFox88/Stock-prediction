# Stock prediction

## Description

This repository does stock prediciton on price data from [AAPL.csv](./AAPL.csv). We expirament with price prediciton using various trained models from sklearn. This project prints out an error plot and error perfromance stats for each trained model.

### Prediction models used:

- lag (previous day's price)
- linear regression
- ridge regresssion
- gradient boosting trees

Ridge regresssion was found to have the least Mean Squared Error, and thus best prediciton performance. Price predictions via ridge regresssion are graphed at the end of the program.

![predictions](https://i.imgur.com/L0CA9iA.jpg)
 
## Dependencies

``pip install matplotlib``

``pip install sklearn``
