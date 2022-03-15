from scripts.definitions import compute_domega1dt, compute_domega2dt, convert
from scipy.integrate import odeint

import numpy as np


TIME    = 10    # Default time in seconds for the simulation


def model(z, t) -> list:
    # System of first order ODE's
    dtheta1dt = z[2]
    dtheta2dt = z[3]
    domega1dt = compute_domega1dt(z)
    domega2dt = compute_domega2dt(z)
    return [dtheta1dt, dtheta2dt, domega1dt, domega2dt]


def run_simulation(z0:  list=[0, 0, 2, 2], tf:   int=TIME, convert: bool=False) -> list:
    # Run double pendulum simulation for a set of initial conditions
    # z0 -> [theta1, theta2, omega1, omega2]

    z   = odeint(model, z0, np.linspace(0, tf, 30*tf))

    if convert:
        pend1, pend2 = convert(z)

        pend1 = [[val[0] for val in pend1], [val[1] for val in pend1]]
        pend2 = [[val[0] for val in pend2], [val[1] for val in pend2]]

        return pend1, pend2
    
    else:
        return z