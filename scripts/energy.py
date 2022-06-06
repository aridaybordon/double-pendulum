from numpy import sqrt, asarray
from scripts.definitions import kinetic_energy, potential_energy


def compute_factor_k(z, E0) -> float:
    "Compute the proportionality factor to guarantee energy conservation"
    Tt, Vt = kinetic_energy(z), potential_energy(z)
    return sqrt((E0 - Vt) / Tt)


def correct_configuration(z, E0) -> list:
    """
    Return the corrected congifuration with the appropiate factor for the velocities
    to guarantee conservation of energy.
    
    """
    theta1, theta2, omega1, omega2 = z
    k = compute_factor_k(z, E0)
    return [[theta1, theta2, k * omega1, k * omega2]]
