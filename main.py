import numpy as np
import json

from scripts.model import create_model, train_model
from tensorflow.keras.utils import normalize


def main() -> None:
    # Create and train model
    model = create_model()
    #train_model(model)
    model.load_weights('checkpoint/cp.ckpt')

    # Compute NN prediction for the double pendulum move
    z       = np.zeros(300, dtype=object)
    z[0]    = np.array([0, 0, 3, -2])      # Initial conditions

    # Use every last iteration to predict next movement step
    for i in range(1, 300):
        z[i] = model.predict(z[i-1])
    
    # Save prediction as json
    with open('data/nn_simulation.json', 'w') as f:
        json.dump({key: value for key, value in zip(range(300), z)}, f)


if __name__ == '__main__':
    main()