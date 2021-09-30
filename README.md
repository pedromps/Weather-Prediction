# Weather-Prediction
 
This is a small test with LSTM to try and predict minimum and maximum temperatures for each day.


It works by training an LSTM network having as input the atmospheric readings of several features from the 3 days before (the amount of days is up to the operator) and the minimum and maximum temperature of the day after as the features to predict.


Right now, with the presented model (which has the results for the test set in the results folder), it has an MAE of around 1.81 degrees Celsius across both temperatures. The MAE for the minimum temperatures is 1.27 degress and for maximum temperatures is 2.34 degrees.