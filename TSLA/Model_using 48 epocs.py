import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# importing the required python packages.

# loading the file and pre-processing the data.
file = "..\..\Stock Dataset\TSLA.csv"
stock = pd.read_csv(file)
stock['Date'] = pd.to_datetime(stock['Date'])
stock = stock.set_index('Date')
print(stock.head())


# creating a plot of the historical data of the given stock.
plt.figure(figsize = (16,8))
plt.title("Stock Price History of Tesla Stock")
plt.plot(stock['Close'])
plt.xlabel('Date')
plt.ylabel('Price')

# setting the split-size of stock data to 80% to tdl.
close_prices = stock['Close']
values = close_prices.values
tdl = math.ceil(len(values)*0.8)

# scaling the stock prices to a value between 0 and 1.
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))


print(values,"\n")
print(scaled_data)
# printing the scaled stock data

# splitting into training and testing datasets(80%-20%), where 'training' has the rows from 0 to tdl(80%) and 'test_data' is from tdl to last row(20%) of stock price.
training = scaled_data[0: tdl, :]

x_train = []
y_train = []

for i in range(80, len(training)):
    x_train.append(training[i - 80:i, 0])
    y_train.append(training[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[tdl - 80:, :]
x_test = []
y_test = values[tdl:]

for i in range(80, len(test_data)):
    x_test.append(test_data[i - 80:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# intializing the model structure
model = keras.Sequential()
model.add(layers.LSTM(100,return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(layers.GRU(100, return_sequences = False))
model.add(layers.Dense(100))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
print(model.summary())

# compiling and training the model on the training dataset.
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size = 1 ,epochs = 48, verbose = 1)

# running the model prediction on the testing dataset, and printing the RMSE value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print("Root Mean Square Error = ", rmse)

# to generate the expected values vs prediction graph, the appropriate values are taken from the stock dataset, for computing the r2 score.
data = stock.filter(['Close'])
train = data[:tdl]
validation = data[tdl:]
validation['Predictions'] = predictions

y_true = y_test
y_pred = validation['Predictions']

# Calculating the r2 score of the model
from sklearn.metrics import r2_score
score = r2_score(validation['Close'],validation['Predictions'])
print("The accuracy of the model is {}%".format(round(score, 2) *100))

# plotting the comparison graph
plt.figure(figsize=(16,8))
plt.title('Tesla Stock Prediction using 48 epocs.')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.savefig("..\..\Stock Prediction\TSLA\model_48_epocs.png")# chnge file name
plt.show()

