import numpy as np
import tensorflow as tf

from scripts.definitions import total_energy
from scripts.energy import correct_configuration
from scripts.data import load_training_data, generate_training_data

from keras.models import Sequential
from keras.layers import Dense

import json


def create_model():
    # Create NN model (4 inputs - 7 hidden layers (1000 neurons/layer) - 4 outputs)
    model = Sequential()

    model.add(Dense(units=1000, activation='relu', input_dim=4))
    model.add(Dense(units=2000, activation='relu'))
    model.add(Dense(units=2000, activation='relu'))
    model.add(Dense(units=2000, activation='relu'))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dense(units=4, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model


def load_model():
    model = create_model()
    model.load_weights("checkpoint/cp.ckpt")

    return model


def train_model(create_data: bool = True, load_weights: bool = True) -> None:
    model = create_model()

    if load_weights:
        model.load_weights("checkpoint/cp.ckpt")
    if create_data:
        generate_training_data()

    # Load and normalize training data
    inp_train, out_train = load_training_data()

    # Define checkpoint path
    checkpoint_path = "checkpoint/cp.ckpt"

    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=0)

    # Fit model
    model.fit(inp_train,
              out_train,
              epochs=10,
              batch_size=128,
              callbacks=[cp_callback])

    return model


def make_and_save_prediction(model, z0: list):
    # Compute NN prediction for the double pendulum move from initial condition z0
    pred_size = 300
    z = np.zeros(pred_size, dtype=object)
    z[0] = z0

    E0 = total_energy(z0)

    # Use every last iteration to predict next movement step
    for i in range(1, pred_size):
        #z[i] = correct_configuration(model.predict([z[i - 1]]), E0)
        pred = model.predict([z[i-1]])
        z[i] = np.asarray(correct_configuration(pred[0], E0))
        print(f"Creating movement {(i/(pred_size-1)):.2%}", end='\r')

    # Save prediction as json
    with open('data/nn_simulation.json', 'w') as f:
        json.dump({key: value.tolist()[0]
                   for key, value in enumerate(z[1:])}, f)
