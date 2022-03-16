import numpy as np
from numpy.random import random

from scripts.simulation import run_simulation

import json


def generate_data() -> list:
    # Generate input and output raw data
    z0           = [2*np.pi*random(), 2*np.pi*random(), 5*(2*random()-1), 5*(2*random()-1)]
    z            = run_simulation(z0=z0, tf=10000)
    return z


def preprocessing(z) -> list:
    # Preprocess data -> Ensures angle is in range(0, 2*np.pi)
    return [[np.mod(z_pre[0], 2*np.pi), np.mod(z_pre[1], 2*np.pi), z_pre[2], z_pre[3]] for z_pre in z]


def save_data(filename: str) -> None:
    # Save training data in json
    with open(f'data/{filename}.json', 'w') as f:

        print('=====================================')
        print('Generating and saving simulation data')
        print('=====================================\n')
        
        z = generate_data()
        print('Data generated!\n')

        json.dump({key: value for key, value in enumerate(preprocessing(z))}, f)


def load_data(filename) -> list:
    # Load input and output training data
    with open(f'data/{filename}.json', 'r') as f:
        data = list(json.load(f).values())
    
    inp, out = data[:-1], data[1:]

    return inp, out
