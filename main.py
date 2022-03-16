import numpy as np
import json

from scripts.model import create_model, train_model


def main() -> None:
    # ==================================================================
    # Load model, train it with newly created data and simulate movement
    # ==================================================================

    # Create and train model
    model = create_model()
    model.load_weights("checkpoint/cp.ckpt")
    train_model(model)

    # Compute NN prediction for the double pendulum move
    pred_size   = 300 
    z           = np.zeros(pred_size, dtype=object)
    z[0]        = [2.5, 1.4, 1, -2]      # Initial conditions

    # Use every last iteration to predict next movement step
    for i in range(1, pred_size):
        z[i] = model.predict([z[i-1]])
        print(f"Creating movement {(i/pred_size):.2%}", end='\r')
    

    # Save prediction as json
    with open('data/nn_simulation.json', 'w') as f:
        json.dump({key: np.asarray(value).tolist() for key, value in enumerate(z)}, f)


if __name__ == '__main__':
    main()