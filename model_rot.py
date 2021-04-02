# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:21:28 2020

@author: kreym
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import model_from_json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

L = 5
size_inputs = (2*L-1)**4

# %% Load datasets
X = np.load('acs_10000.npy') # Input

y = np.load('acs_neigh_10000.npy') # Output

# %% define the keras model
model = keras.Sequential()
model.add(layers.Embedding(input_dim=(2*L-1)**4, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense((2*L-1)**4))

model.summary()

# %% Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mse'])

sizes = [9500]
for i in range(0, len(sizes)):
    print(i)
    sizedata = sizes[i]
    model.fit(X[0:sizedata, :], y[0:sizedata, :], epochs=100, batch_size=1000)
    # serialize model to JSON
    model_json = model.to_json()
    with open(f"model_acs2acs_neigh_{sizedata}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"model_acs2acs_neigh_{sizedata}.h5")
    print("Saved model to disk")
    
    
    # # load json and create model
    # json_file = open(f"model_well_separated_same_gamma_{sizes[i-1]}.json", 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(f"model_well_separated_same_gamma_{sizes[i-1]}.h5")
    # print("Loaded model from disk")
    
    # # %% Compile the model
    # loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # %% fit the keras model on the dataset
    # loaded_model.fit(X[sizes[i-1]:sizes[i], :], y[sizes[i-1]:sizes[i], :], epochs=1000, batch_size=100)
    
    # # evaluate the keras model
    # _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))
    
    prediction = model.predict(X[39000: , :])
    vals = y[39500: , :]
    
    err = np.sqrt(np.sum((prediction[:, :] - vals[:, :])**2, axis=1)) / np.sqrt(np.sum(vals[:, 1:]**2, axis=1))
    

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(np.reshape(prediction[0, 1:], (L, L)), cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(np.reshape(vals[0, 1:], (L, L)), cmap='gray')
    # plt.show()
    
    # # serialize model to JSON
    # model_json = loaded_model.to_json()
    # with open(f"model_well_separated_same_gamma_{sizes[i]}.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # loaded_model.save_weights(f"model_well_separated_same_gamma_{sizes[i]}.h5")
    # print("Saved model to disk")

# # load json and create model
# json_file = open("model_well_separated_same_gamma_1000.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model_well_separated_same_gamma_1000.h5")
# print("Loaded model from disk")
# loaded_prediction = loaded_model.predict(X[sizedata:, :])
# loaded_err = np.sqrt(np.sum((loaded_prediction[:, 1:] - vals[:, 1:])**2, axis=1)) / np.sqrt(np.sum(vals[:, 1:]**2, axis=1))



