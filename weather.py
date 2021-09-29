# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from sklearn.model_selection import train_test_split

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


lookback = 144*3 # observations look back in 5 days
# step = 6 # one data point per hour

# temperature is the target
target = float_data[:, 1]
float_data = np.delete(float_data, 1, axis=1)


# split a univariate sequence into samples
def split_sequence(sequence, target, n_steps, lookback):
    X, y = list(), list()
    for i in range(0, sequence.shape[0], lookback):
		# find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the sequence
        if end_ix > sequence.shape[0]-lookback:
            break
		# gather input and output parts of the pattern
        # minimum temperature of the NEXT day to the patterns
        seq_x, min_y, max_y = sequence[i:end_ix], np.min(target[end_ix:end_ix+lookback]), np.max(target[end_ix:end_ix+lookback])
        X.append(seq_x)
        y.append((min_y, max_y))
    return np.array(X), np.array(y)
 
# choose a number of time steps
lookback = 144 # <- 1 day
# choose size of each sequence
seq_size = 144*3 # <- 3 days
# split into samples
X, y = split_sequence(float_data, target, seq_size, lookback)

# each X has 3 days, each Y is the forecast of the day after those 3.
# shuffle false so that the data is sequential
# the goal is to train with the first 80% of days, validate with the following 10% and test with the rest
# 0.9 as it is train + val
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, shuffle = False)
# 10% of the total data -> 0.1/0.9 = 1/9 of the train data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 1/9, shuffle = False)
    
#RMSE loss function 
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

model = Sequential()
model.add(LSTM(128, activation = 'tanh', dropout = 0.1, input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences = True))
model.add(LSTM(32, activation = 'tanh'))
model.add(Dense(y_train.shape[1], activation = 'relu'))
model.compile(optimizer = "adam", loss = root_mean_squared_error)
model.summary()
history = model.fit(x_train, y_train, epochs = 100, verbose = 1, batch_size = 128, validation_data = (x_val, y_val))

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.grid()
plt.title('Training and validation loss')
plt.legend()
plt.show()

temp_pred = model.predict(x_test)
# MAE = np.mean(np.abs(y_test.ravel()-temp_pred.ravel()))
# denorm MSE = c^2 * MSE, c = (max-min)
denorm_MAE = np.mean((norm_max[1]-norm_min[1])*np.abs(y_test.ravel()-temp_pred.ravel()))
print("MAE = {:.2f}".format(denorm_MAE) + " degres")

denorm = temp_pred*(norm_max[1] - norm_min[1]) + norm_min[1]
denorm_y = y_test*(norm_max[1] - norm_min[1]) + norm_min[1]
# plotting predictions against the actual values
fig, axs = plt.subplots(1, 2, sharex = True)
axs[0].plot(denorm[:,0])
axs[0].plot(denorm_y[:,0])
axs[0].grid()
axs[0].legend(["Min T", "Predicted Min T"], loc = 'best')
axs[1].plot(denorm[:,1])
axs[1].plot(denorm_y[:,1])
axs[1].grid()
axs[1].legend(["Max T", "Predicted Max T"], loc = 'best')
plt.tight_layout()


# DEPRECATED
# def train_test_val_split(data, y, lookback, step, mode):
#     # training data will be derived from the first sequences, validation from the 
#     # sequences following them and test following validation
#     size = len(data)
#     train_size = [0, int(0.8*size)-int(0.8*size)%lookback] #<- 408 sequences
#     val_size = [train_size[1], train_size[1]+int(0.1*size)-int(0.1*size)%lookback]
#     test_size = [val_size[1], val_size[1]+int(0.1*size)-int(0.1*size)%lookback]
    
    
#     st = train_size[1]-train_size[0]
#     aux = np.copy(data[train_size[0]:train_size[1]])
#     aux_y = np.copy(y[train_size[0]:train_size[1]])
#     x_train = np.reshape(aux[::step], (int(st/lookback), int(lookback/step), data.shape[1]))
#     y_train = np.reshape(aux_y[::step], (int(st/lookback), int(lookback/step), 1))
    
#     sv = val_size[1]-val_size[0]
#     aux = np.copy(data[val_size[0]:val_size[1]])
#     aux_y = np.copy(y[val_size[0]:val_size[1]])
#     x_val = np.reshape(aux[::step], (int(sv/lookback), int(lookback/step), data.shape[1]))
#     y_val = np.reshape(aux_y[::step], (int(sv/lookback), int(lookback/step), 1))
    
#     stt = test_size[1]-test_size[0]
#     aux = np.copy(data[test_size[0]:test_size[1]])
#     aux_y = np.copy(y[test_size[0]:test_size[1]])
#     x_test = np.reshape(aux[::step], (int(stt/lookback), int(lookback/step), data.shape[1]))
#     y_test = np.reshape(aux_y[::step], (int(stt/lookback), int(lookback/step), 1))
    
#     if mode == "max":
#         y_train = np.max(y_train, axis = 1)
#         y_val = np.max(y_val, axis = 1)
#         y_test = np.max(y_test, axis = 1)
        
#     if mode == "min":
#         y_train = np.min(y_train, axis = 1)
#         y_val = np.min(y_val, axis = 1)
#         y_test = np.min(y_test, axis = 1)
        
#     if mode == "minmax" or mode == "maxmin":
#         y_train = np.concatenate([np.min(y_train, axis = 1), np.max(y_train, axis = 1)], axis = 1) 
#         y_val = np.concatenate([np.min(y_val, axis = 1), np.max(y_val, axis = 1)], axis = 1) 
#         y_test = np.concatenate([np.min(y_test, axis = 1), np.max(y_test, axis = 1)], axis = 1)
        
#     return x_train, x_test, x_val, y_train, y_test, y_val


# x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(float_data, target, lookback, step, "minmax")
