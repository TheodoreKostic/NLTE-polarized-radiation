"""Shared line-profile functions used by opacity and radiation tensors."""
import numpy as np
from scipy.special import wofz

NU_LARMOR = 1.3996e6  # Larmor frequency per gauss [Hz/G]; single source for all Hanle/Zeeman terms


def voigt_function(x, a):
    """Complex Voigt function w(x + i a) / sqrt(pi), with x in Doppler units."""
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    if np.any(a < 0.0):
        raise ValueError("Voigt damping parameter a must be non-negative")
    return wofz(x + 1j * a) / np.sqrt(np.pi)
