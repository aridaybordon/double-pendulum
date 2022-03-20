import numpy as np

from numpy.random import random
from scripts.simulation import run_simulation

import json


def generate_data(time: float=1) -> list:
    # Generate input and output raw data
    z0 = [2*np.pi*random(), 2*np.pi*random(), 2 * (2*random()-1), 2*(2*random()-1)]
    z = run_simulation(z0=z0, tf=time)
    return z


def preprocessing(z) -> list:
    # Preprocess data -> Ensures angle is in range(0, 2*np.pi)
    return [[np.mod(z_pre[0], 2*np.pi), np.mod(z_pre[1], 2*np.pi), z_pre[2], z_pre[3]] for z_pre in z]


def generate_training_data(verbose=True) -> None:
    # Save training data in json
    n_iter = 100000 # -> Generates 1 day of training data
    test_inp, test_out = [], []

    if verbose:
        print("\nGenerating training data:")

    for _ in range(n_iter):
        z = generate_data()

        [test_inp.append(inp.tolist()) for inp in z[:-1]]
        [test_out.append(out.tolist()) for out in z[1:]]
        
        if verbose:
            print(f"Completed {(_+1)/n_iter:.2%}", end="\r")

    with open("data/training_output.json", "w") as f:
        json.dump({key: value for key, value in enumerate(test_out)}, f)
    
    with open("data/training_input.json", "w") as f:
        json.dump({key: value for key, value in enumerate(test_inp)}, f)


def load_training_data() -> list:
    # Load input and output training data
    with open("data/training_output.json", "r") as f:
        out = list(json.load(f).values())
    
    with open("data/training_input.json", "r") as f:
        inp = list(json.load(f).values())

    return inp, out
