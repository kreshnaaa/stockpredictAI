import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv("C:/Users/ADMIN/Desktop/data_pipeline/all_stocks_5yr.csv")

# Filter the data for the stock of interest
stock_name = "AAPL"  # Replace with the stock name you want to predict
df = df[df['Name'] == stock_name]

# Convert 'date' to datetime with the correct format and set it as the index
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')  # Use '%d-%m-%Y' format
df.set_index('date', inplace=True)


# # Convert 'date' to datetime and set it as the index
# df['date'] = pd.to_datetime(df['date'])
# df.set_index('date', inplace=True)

# Use 'close' or 'adj_close' as the target variable
target_column = 'close'  # Change to 'adj_close' if needed
df = df[[target_column]]

# Scale the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Define a function to create time series data
def create_time_series_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# Define the look-back window (number of time steps to use for prediction)
look_back = 10

X, y = create_time_series_data(df_scaled, look_back)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and train a time series forecasting model (LSTM)
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Predict future stock prices
y_pred = model.predict(X_test)

# Inverse transform the scaled values to get actual prices
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the actual vs. predicted stock prices
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()









# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import mean_squared_error



# df=pd.read_csv("C:/Users/ADMIN/Desktop/data_pipeline/all_stocks_5yr.csv")

# print(df.head())
# df.info()

# print(df.columns)

# # Remove missing values
# df = df.dropna()


# # Drop non-numeric columns that you don't intend to use in your model
# df = df.drop(['date', 'Name'], axis=1)
# # Scale the data

# scaler = MinMaxScaler()
# df_scaled = scaler.fit_transform(df)

# # Split the data into training and testing sets

# X_train, X_test, y_train, y_test = train_test_split(df_scaled[:, :-1], df_scaled[:, -1], test_size=0.2, random_state=42)

# #Create the neural network
# model = Sequential()
# model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='linear'))

# #Train the neural network

# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# #Evaluate the neural network:

# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)