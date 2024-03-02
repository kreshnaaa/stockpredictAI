import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import math
import yfinance as yf

# Define the stock symbol, start date, and end date
symbol = 'AAPL'
start_date = '2012-01-01'
end_date = '2023-10-27'

# Fetch the stock data
df = yf.download(symbol, start=start_date, end=end_date)


# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])

print(data)

# Convert the dataset into a numpy array
dataset = data.values

print(dataset)

# Get the number of rows to train the model on (80% of the data)
training_data_len = math.ceil(len(dataset) * 0.8)

print(training_data_len)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset) # Here totally row values i.e 2974 rows

# Create the training data set
train_data = scaled_data[0:training_data_len, :] #Here dataset in slace has taken [0:2380,:] in 2D array means [rows range,allcolumns]

print(train_data)

# Split the data into X_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #Here reshape(2320,60,1)

print(x_train.shape)

# Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
# Create a copy of the 'valid' DataFrame before adding the 'Predictions' column
valid = valid.copy()
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the valid and predictions
print(valid)

# Get the quote for the available date day
start_date = '2023-10-25'
end_date = '2023-10-25'
apple_quote1 = yf.download(symbol, start=start_date, end=end_date)
# Check if data is available for the selected date
if not apple_quote1.empty:
    print("Closing Price on 2023-10-25:", apple_quote1['Close'].values[0])
else:
    print("Data not available for 2023-10-25")

# Use the model to predict the next day's price
last_60_days = data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print("Predicted Price on 2023-10-28:", pred_price[0][0])
