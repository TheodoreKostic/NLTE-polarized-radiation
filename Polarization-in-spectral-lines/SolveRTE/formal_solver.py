"""Formal solution helpers for polarized radiative transfer."""
import numpy as np
from scipy.linalg import expm


def propagation_matrix(coefficients):
    """Build the Stokes propagation matrix from eta and magneto-optical terms.

    ``coefficients`` is ordered as
    (eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V), and the Stokes vector
    is ordered (I, Q, U, V).
    """
    eta_i, eta_q, eta_u, eta_v, rho_q, rho_u, rho_v = np.asarray(
        coefficients, dtype=float
    )
    return np.array(
        [
            [eta_i, eta_q, eta_u, eta_v],
            [eta_q, eta_i, rho_v, -rho_u],
            [eta_u, -rho_v, eta_i, rho_q],
            [eta_v, rho_u, -rho_q, eta_i],
        ],
        dtype=float,
    )


def propagate_constant_slab(emissivity, coefficients, distance, incident):
    """Propagate a Stokes vector through a homogeneous slab exactly.

    Solves dS/ds = emissivity - K @ S over a non-negative geometric path
    length ``distance``. Coefficients and emissivity are constant over the
    slab. An augmented matrix exponential handles singular K without requiring
    its inverse.
    """
    emissivity = np.asarray(emissivity, dtype=float)
    incident = np.asarray(incident, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if emissivity.shape != (4,):
        raise ValueError(f"emissivity must have shape (4,), got {emissivity.shape}")
    if incident.shape != (4,):
        raise ValueError(f"incident must have shape (4,), got {incident.shape}")
    if coefficients.shape != (7,):
        raise ValueError(
            f"coefficients must have shape (7,), got {coefficients.shape}"
        )
    if not np.all(np.isfinite(emissivity)):
        raise ValueError("emissivity must contain only finite values")
    if not np.all(np.isfinite(incident)):
        raise ValueError("incident must contain only finite values")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must contain only finite values")
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be finite and non-negative")

    matrix = propagation_matrix(coefficients)
    augmented = np.zeros((5, 5), dtype=float)
    augmented[:4, :4] = -matrix
    augmented[:4, 4] = emissivity
    result = expm(augmented * distance) @ np.append(incident, 1.0)
    return result[:4]
