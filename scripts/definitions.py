from numpy import sin, cos

# Vector z = [theta1, theta2, omega1, omega2]
# with omega = dtheta/dt

EPS = 1     # m1/m2
ETA = 1     # l1/l2

g   = 10    # g/l2

# Useful definitions from https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
def alpha1(z):
    return 1/(ETA * (1 + EPS)) * cos(z[0] - z[1])


def alpha2(z):
    return ETA * cos(z[0] - z[1])


def f1(z):
    return - 1/(ETA * (1 + EPS)) * z[3]**2 * sin(z[0] - z[1]) - g/ETA * sin(z[0])


def f2(z):
    return ETA * z[2]**2 * sin(z[0] - z[1]) - g * sin(z[1])


# Variations of omega1 and omega2
def compute_domega1dt(z):
    return 1 / (1 - alpha1(z)*alpha2(z)) * (f1(z) - alpha1(z) * f2(z))


def compute_domega2dt(z):
    return 1 / (1 - alpha1(z)*alpha2(z)) * (f2(z) - alpha2(z) * f1(z))


# Conversion (theta1, theta2) -> (x, y)
def convert(z):
    pend1 = [[ETA*sin(i[0]), -ETA*cos(i[0])] for i in z]
    pend2 = [[prev[0] + ETA*sin(i[1]), prev[1] - ETA*cos(i[1])] for prev, i in zip(pend1, z)]

    return pend1, pend2
