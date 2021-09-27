# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed

# code adapted from F. Chollet's book Deep Learning With Python
fname = 'jena_climate_2009_2016.csv'
f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
    
norm_max = np.nanmax(float_data, axis = 0)
norm_min = np.nanmin(float_data, axis = 0)
float_data = (float_data-norm_min)/(norm_max-norm_min)


lookback = 720 # observations look back in 10 days
step = 6 # one data point per hour
delay = 144 # 24h
batch_size = 128

# temperature is the target
target = float_data[:, 2]
float_data = np.delete(float_data, 2, axis=1)

def train_test_val_split(data, y, lookback, step):
    # training data will be derived from the first sequences, validation from the 
    # sequences following them and test following validation
    size = len(data)
    train_size = [0, int(0.8*size)-int(0.8*size)%lookback] #<- 408 sequences
    val_size = [train_size[1], train_size[1]+int(0.1*size)-int(0.1*size)%lookback]
    test_size = [val_size[1], val_size[1]+int(0.1*size)-int(0.1*size)%lookback]
    
    
    st = train_size[1]-train_size[0]
    aux = np.copy(data[train_size[0]:train_size[1]])
    aux_y = np.copy(y[train_size[0]:train_size[1]])
    x_train = np.reshape(aux[::step], (int(st/lookback), int(lookback/step), data.shape[1]))
    y_train = np.reshape(aux_y[::step], (int(st/lookback), int(lookback/step), 1))
    
    sv = val_size[1]-val_size[0]
    aux = np.copy(data[val_size[0]:val_size[1]])
    aux_y = np.copy(y[val_size[0]:val_size[1]])
    x_val = np.reshape(aux[::step], (int(sv/lookback), int(lookback/step), data.shape[1]))
    y_val = np.reshape(aux_y[::step], (int(sv/lookback), int(lookback/step), 1))
    
    stt = test_size[1]-test_size[0]
    aux = np.copy(data[test_size[0]:test_size[1]])
    aux_y = np.copy(y[test_size[0]:test_size[1]])
    x_test = np.reshape(aux[::step], (int(stt/lookback), int(lookback/step), data.shape[1]))
    y_test = np.reshape(aux_y[::step], (int(stt/lookback), int(lookback/step), 1))
    
    return x_train, x_test, x_val, y_train, y_test, y_val


x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(float_data, target, lookback, step)


model = Sequential()
model.add(LSTM(128, activation = 'relu', input_shape = (x_train.shape[1], x_train.shape[2])))
# model.add(TimeDistributed(Dense(train.shape[1], activation = 'relu')))
model.add(Dense(x_train.shape[1]))
model.compile(optimizer = "adam", loss = 'mse')
model.summary()
history = model.fit(x_train, y_train, epochs = 40, verbose = 1, batch_size = 256, validation_data = (x_val, y_val))

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

temp_pred = model.predict(x_test)
MSE = np.mean((y_test.ravel()-temp_pred.ravel())**2)

