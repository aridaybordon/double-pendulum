import numpy as np
from numpy.random import random

from scripts.simulation import run_simulation

import json


def generate_data() -> list:
    # Generate input and output data
    z0           = [2*np.pi*random(), 2*np.pi*random(), 5*(2*random()-1), 5*(2*random()-1)]
    z            = run_simulation(z0=z0, tf=100)
    return z


def save_data(filename: str) -> None:
    # Save training data in json
    with open(f'data/{filename}.json', 'w') as f:

        print('=====================================')
        print('Generating and saving simulation data')
        print('=====================================\n')
        
        z = generate_data()
        print('Data generated!')

        json.dump({key: value.tolist() for key, value in zip(range(len(z)), z)}, f)


def load_data(filename) -> list:
    # Load input and output training data
    with open(f'data/{filename}.json', 'r') as f:
        data = list(json.load(f).values())
    
    inp, out = data[:-1], data[1:]

    return inp, out
