import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
import math
import yfinance as yf
symbol = 'AAPL'
start_date = '2012-01-01'
end_date = '2023-10-27'

# Fetch the stock data
df = yf.download(symbol, start=start_date, end=end_date)

# Print the first few rows of the data
print(df.tail(5))


print(df.shape)

#visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('close price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

#Create a new dataframe with only the 'Close column'

data=df.filter(['Close'])

#convert the dataset into a numpy array

dataset=data.values

#Get the number of rows to train the model on

training_data_len=math.ceil(len(dataset)*.8) #80 %

#Scale the data 

scaler=MinMaxScaler(feature_range=(0,1))

scaled_data=scaler.fit_transform(dataset)

#create the training data set

#create the scaled training data set

train_data=scaled_data[0:training_data_len,:]

#Split the data into X_train and y_train data sets

x_train=[]

y_train=[]

#Creating a data structure with 60 timesteps and 1 output

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

    if i<=61:
        print(x_train)
        print(y_train)

#convert thr x_train and y_train to numppy array

x_train,y_train=np.array(x_train),np.array(y_train)

#reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Initialising the RNN

model=Sequential()

#Adding the first LSM Layer and same Dropout regularisation

model.add(LSTM(units=50,return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(Dropout(.2))

#Adding a second LSTM Layer and same Dropout regularisation

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(.2))

#Adding a third LSTM Layer and same Dropout regularisation

model.add(LSTM(50,return_sequences=True))
model.add(Dropout(.2))

#Adding a fourth LSTM Layer and same Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(.2))

#Adding the output layer
model.add(Dense(units=1))

#Compile the RNN

model.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNN to the Training set

model.fit(x_train,y_train,batch_size=32,epochs=10)

#Part 3:Matching the predictions and visualising the results


#create a new array containing scaled values from index 2320 to 

test_data=scaled_data[training_data_len-60:,:]

#create the data sets x_test and y_test
 
x_test=[]

y_test=dataset[training_data_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#convert the data to a numpy

x_test=np.array(x_test)

#reshape the data

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get the models predicted price values

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

#Get the root mean squared error(RMSE)

rmse=np.sqrt(np.mean(predictions-y_test)**2)

print(rmse)

#plot the data

train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()

#Show the valid and pridictiona

print(valid)

#Get the quote

apple_quote = yf.download(symbol, start=start_date, end=end_date)

#Create a new data frame

new_df=df.filter(['Close'])

#Get the last 60 days closing price and convert the dataframe to array

last_60_days=new_df[-60:].values

#Scale the data to be values between 0 and 1

last_60_days_scaled=scaler.transform(last_60_days)

#Create an empty_lis

X_test=[]

#append ten past 60 days

X_test.append(last_60_days_scaled)

# convert the X_test data set to a numpy array

X_test=np.array(X_test)

#Reshape the data

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#Get the Predicted scaled price

pred_price=model.predict(X_test)

#undo the scaling

pred_price=scaler.inverse_transform(pred_price)

print(pred_price)

#Get the quote

start_date='2023-10-28'

end_date='2023-10-28'

apple_quote1 = yf.download(symbol, start=start_date, end=end_date)
print(apple_quote1['Close'])
 
print()

