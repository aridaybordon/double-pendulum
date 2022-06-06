from numpy import sin, cos

# Vector z = [theta1, theta2, omega1, omega2]
# with omega = dtheta/dt

EPS = 1  # m1/m2
ETA = 1  # l1/l2

g = 10  # g/l2


# Useful definitions from https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
def alpha1(z):
    return 1 / (ETA * (1 + EPS)) * cos(z[0] - z[1])


def alpha2(z):
    return ETA * cos(z[0] - z[1])


def f1(z):
    return -1 / (ETA *
                 (1 + EPS)) * z[3]**2 * sin(z[0] - z[1]) - g / ETA * sin(z[0])


def f2(z):
    return ETA * z[2]**2 * sin(z[0] - z[1]) - g * sin(z[1])


# Energy constraints
def kinetic_energy(z):
    "Compute the kinetic energy for a given configuration z"
    theta1, theta2, omega1, omega2 = z
    return 1 / 2 * EPS * ETA**2 * omega1**2 + 1 / 2 * (
        ETA**2 * omega1**2 + omega2**2 +
        2 * ETA * omega1 * omega2 * cos(theta1 - theta2))


def potential_energy(z):
    "Compute the potential energy for a given configuration z"
    theta1, theta2, omega1, omega2 = z
    return -g * ((1 + EPS) * ETA * cos(theta1) + cos(theta2))


def total_energy(z) -> float:
    "Compute energy for a given configuration "
    return kinetic_energy(z) + potential_energy(z)


# Variations of omega1 and omega2
def compute_domega1dt(z):
    "Compute the variation of theta1"
    return 1 / (1 - alpha1(z) * alpha2(z)) * (f1(z) - alpha1(z) * f2(z))


def compute_domega2dt(z):
    "Compute the variation of theta2"
    return 1 / (1 - alpha1(z) * alpha2(z)) * (f2(z) - alpha2(z) * f1(z))


# Conversion (theta1, theta2) -> (x, y)
def convert(z):
    "Convert from polar to cartesian coordinates"
    pend1 = [[ETA * sin(i[0]), -ETA * cos(i[0])] for i in z]
    pend2 = [[prev[0] + ETA * sin(i[1]), prev[1] - ETA * cos(i[1])]
             for prev, i in zip(pend1, z)]

    return pend1, pend2
