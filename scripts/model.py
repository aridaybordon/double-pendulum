import numpy as np
import tensorflow as tf

from scripts.data import load_data, generate_training_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import json


def create_model():
    # Create NN model (4 inputs - 2 hidden layers (5 neurons/layer) - 4 outputs)
    model = Sequential()
    
    model.add(Dense(units=20, activation='sigmoid', input_dim=4))
    model.add(Dense(units=20, activation='sigmoid'))
    model.add(Dense(units=20, activation='sigmoid'))
    model.add(Dense(units=4, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model


def train_model(model, create_data: bool=True) -> None:
    # Create training data
    if create_data:
        generate_training_data('train')

    # Load and normalize training data
    inp_train, out_train    = load_data('train')

    # Define checkpoint path
    checkpoint_path = "checkpoint/cp.ckpt"

    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        save_weights_only=True,
        verbose=1
        )

    # Fit model
    model.fit(inp_train, out_train, epochs=1, batch_size=1, callbacks=[cp_callback])


def training_routine(load_weights=True):
    model = create_model()
    
    if load_weights:
        model.load_weights("checkpoint/cp.ckpt")
    
    train_model(model, create_data=False)

    return model


def make_and_save_prediction(model, z0: list):
    # Compute NN prediction for the double pendulum move from initial condition z0
    pred_size   = 300 
    z           = np.zeros(pred_size, dtype=object)
    z[0]        = z0

    # Use every last iteration to predict next movement step
    for i in range(1, pred_size):
        z[i] = model.predict([z[i-1]])
        print(f"Creating movement {(i/(pred_size-1)):.2%}", end='\r')
    
    z = z[1:]

    # Save prediction as json
    with open('data/nn_simulation.json', 'w') as f:
        json.dump({key: value.tolist()[0] for key, value in enumerate(z)}, f)
