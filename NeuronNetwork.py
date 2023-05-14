import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.optimizers import Adam

from tensorflow import keras

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


class NeuronNetwork:
    celsius = np.array([0.0, 5.0, 9.0, 29.0, 31.0, 39.0, 41.0])
    fahrenheit = np.array([32.8, 41.0, 48.2, 84.2, 87.8, 102.2, 105.8])

    def create_neuron_network(self):
        model = keras.Sequential()
        dense = Dense(units=1, input_shape=(1,), activation='linear')
        model.add(dense)

        model.compile(loss='mse', optimizer=Adam(0.1))

        history = model.fit(self.celsius, self.fahrenheit, epochs=1000, steps_per_epoch=2, verbose=False)

        plt.plot(history.history['loss'])
        plt.grid(True)
        plt.show()

        print(model.predict([20]))
