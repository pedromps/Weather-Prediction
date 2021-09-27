# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
    
    
mean = float_data[:200000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std

lookback = 720 # observations look back in 10 days
step = 6 # one data point per hour
delay = 144 # 24h
batch_size = 128


def train_test_val_split(data, lookback, step):
    # training data will be derived from the first sequences, validation from the 
    # sequences following them and test following validation
    size = len(data)
    train_size = [0, int(0.7*size)-int(0.7*size)%lookback] #<- 408 sequences
    val_size = [train_size[1], train_size[1]+int(0.2*size)-int(0.2*size)%lookback]
    test_size = [val_size[1], val_size[1]+int(0.1*size)-int(0.1*size)%lookback]
    
    
    st = train_size[1]-train_size[0]
    aux = np.copy(data[train_size[0]:train_size[1]])
    train = np.reshape(aux[::step], (int(st/lookback), int(lookback/step), data.shape[1]))
    
    sv = val_size[1]-val_size[0]
    aux = np.copy(data[val_size[0]:val_size[1]])
    val = np.reshape(aux[::step], (int(sv/lookback), int(lookback/step), data.shape[1]))
    
    stt = test_size[1]-test_size[0]
    aux = np.copy(data[test_size[0]:test_size[1]])
    test = np.reshape(aux[::step], (int(stt/lookback), int(lookback/step), data.shape[1]))

    
    return train, test, val


train, test, val = train_test_val_split(float_data, lookback, step)


model = Sequential()
model.add(LSTM(32, activation = 'relu', input_shape = (train.shape[1], train.shape[2])))
model.add(Dense(1, activation = 'relu'))
model.compile(optimizer = "adam", loss = 'mae')
model.summary()
history = model.fit(train, epochs = 20, verbose = 1)#, validation_data = val)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()