
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
# function to split my data into features and target, shuffle and shift back 1 step: back=1
def mydataset(dataset,back=1):
    X,Y=[],[] # list to hold features and target
    for i in range(len(dataset)-back-1):
        data=dataset[i:(i+back),0]
        X.append(data) # features
        Y.append(dataset[i+back,0]) #target
    return np.array(X),np.array(Y) # returns array of features and targets, here features are past prices and target is future price
np.random.seed(7)
df=pd.read_csv('yahoo2.csv',usecols=[4]) # getting data from the file, we are interested in the 4th column; the closing stocks
dataset=df.values
dataset=dataset.astype('float32') # we convert our data into float type
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset) # Randomly transform our dataset
train_size=int(len(dataset)* 0.80) # getting 80 percent of our data as the training set
test_size=len(dataset)-train_size  # test_size is the dataSize-trainingSize
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]
back=3 # shift our stock prices 3 steps backwards ,so that we can make future forcast
# split the data into test and training set
X_train,Y_train=mydataset(train,back)
X_test,Y_test=mydataset(test,back)
# Reshaping the test and training sets into 1D
X_train=np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test=np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
# LSTM MODEL
model=Sequential() # create an instance of sequential model
model.add(LSTM(10,input_shape=(1,back))) # 1D LSTM model with 10 neurons
model.add(Dropout(0.2)) # dropping out 20 % of my features
model.add(Dense(1))
# Compiling , fitting and finding the accuracy of the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history=model.fit(X_train,Y_train,epochs=10,batch_size=1,verbose=2)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100)) # accuracy =scores[1], loss=scores[0]
# Plotting the loss and accuracy graphs
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

