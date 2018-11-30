import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
date=[] # list to hold dates values
price=[] # List to hold the closing stocks price
# function to read inputs from the data file
def readFile(filename):
    with open(filename,'r') as csvfile:
        csvfileReader=csv.reader(csvfile)
        next(csvfileReader) # making sure all contents of file are read sequentially
        for r in csvfileReader:
            date.append(int(r[0].split('/')[0]))# appending the dates value to a dates list, splitting date form mm/dd/yy
            price.append(float(r[4])) # Closing stock price is in index 4 in the csv file, append it to price list
    return
# fuction to predict prices based on a certain input date, implements Support Vector Machines
# and return possible closing stock prices based on the SVM model
def prediction(dates,prices,x):
    dates=np.reshape(dates,(len(dates),1)) # Reshapind dates array into a 1D array
    lin=SVR(kernel='linear',C=1e3) # implements the linear model
    polyy=SVR(kernel='poly',C=1e3,degree=2) # Polynomial model with deg 2
    rBf=SVR(kernel='rbf',C=1e3,gamma=0.1) # implements the rbf model

     # we will compare the results and see which of the above models gives a best bet
    # Fitting the dates and prices into the model
    lin.fit(dates,prices)
    polyy.fit(dates, prices)
    rBf.fit(dates, prices)
    # Plotting the models
    plt.scatter(dates,prices,color='black',label='Data')
    plt.plot(dates,rBf.predict(dates),color='red',label='RBF')
    plt.plot(dates, lin.predict(dates), color='blue', label='Linear')
    plt.plot(dates, polyy.predict(dates), color='green', label='Polynomial')
    plt.xlabel('dates')
    plt.ylabel('prices')
    plt.legend()
    plt.show()
    return polyy.predict(x)[0],lin.predict(x)[0],rBf.predict(x)[0]
readFile('yahoo2.csv')
predict=prediction(date,price,29)
print(predict)
