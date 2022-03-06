from definitions import compute_domega1dt, compute_domega2dt, convert
from scipy.integrate import odeint

import numpy as np


FRAMES  = 300   # Number of frames
TIME    = 10    # Total time in seconds for the simulation


def model(z, t) -> list:
    # System of first order ODE's
    dtheta1dt = z[2]
    dtheta2dt = z[3]
    domega1dt = compute_domega1dt(z)
    domega2dt = compute_domega2dt(z)
    return [dtheta1dt, dtheta2dt, domega1dt, domega2dt]


def run_simulation(z0 = [0, 0, 3, 3]) -> list:
    # Run double pendulum simulation for a set of initial conditions
    # z0 -> [theta1, theta2, omega1, omega2]

    t   = np.linspace(0, TIME, FRAMES)
    z   = odeint(model, z0, t)

    pend1, pend2 = convert(z)

    pend1 = [[val[0] for val in pend1], [val[1] for val in pend1]]
    pend2 = [[val[0] for val in pend2], [val[1] for val in pend2]]

    return pend1, pend2