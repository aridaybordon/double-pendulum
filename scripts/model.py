import tensorflow as tf

from scripts.definitions import normalize
from scripts.data import load_data, preprocessing, save_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    # Create NN model (4 inputs - 2 hidden layers (5 neurons/layer) - 4 outputs)
    model = Sequential()
    
    model.add(Dense(units=10, activation='relu', input_dim=4))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=4, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model


def train_model(model):
    # Create training data
    #save_data('train')

    # Load and normalize training data
    inp_train, out_train    = load_data('train')
    inp_train, out_train    = normalize(inp_train), normalize(out_train)
    
    # Define checkpoint path
    checkpoint_path = "checkpoint/cp.ckpt"

    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        save_weights_only=True,
        verbose=1
        )

    # Fit model
    model.fit(inp_train, out_train, epochs=5, batch_size=1, callbacks=[cp_callback])
