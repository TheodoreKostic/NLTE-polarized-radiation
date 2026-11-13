"""Formal solution helpers for polarized radiative transfer."""
import numpy as np
from numba import jit
from scipy.linalg import expm


@jit(nopython=True)
def short_characteristics(tau, S, mu, I_boundary):
    """Parabolic short-characteristics sweep for one ray and scalar source.

    Returns the intensity and the diagonal of the Lambda operator at each depth.
    """
    ND = S.shape[0]
    begin = ND - 1
    end = -1
    step = -1
    if mu < 0:
        begin = 0
        end = ND
        step = 1

    I = np.zeros(ND)
    L = np.zeros(ND)
    I[begin] = I_boundary

    for d in range(begin + step, end - step, step):
        delta_u = (tau[d - step] - tau[d]) / mu
        delta_d = (tau[d] - tau[d + step]) / mu
        expd = np.exp(-delta_u)

        if delta_u <= 0.01:
            du = delta_u
            w0 = du * (
                1.0 - du / 2.0 + du**2 / 6.0 - du**3 / 24.0 + du**4 / 120.0
                - du**5 / 720.0 + du**6 / 5040.0 - du**7 / 40320.0 + du**8 / 362880.0
            )
            w1 = du**2 * (
                0.5 - du / 3.0 + du**2 / 8.0 - du**3 / 30.0 + du**4 / 144.0
                - du**5 / 840.0 + du**6 / 5760.0 - du**7 / 45360.0 + du**8 / 403200.0
            )
            w2 = du**3 * (
                1.0 / 3.0 - du / 4.0 + du**2 / 10.0 - du**3 / 36.0 + du**4 / 168.0
                - du**5 / 960.0 + du**6 / 6480.0 - du**7 / 50400.0 + du**8 / 443520.0
            )
        else:
            w0 = 1.0 - expd
            w1 = w0 - delta_u * expd
            w2 = 2.0 * w1 - delta_u**2 * expd

        psi0 = w0 + (
            w1 * (delta_u / delta_d - delta_d / delta_u)
            - w2 * (1.0 / delta_d + 1.0 / delta_u)
        ) / (delta_u + delta_d)
        psiu = (w2 / delta_u + w1 * delta_d / delta_u) / (delta_u + delta_d)
        psid = (w2 / delta_d - w1 * delta_u / delta_d) / (delta_u + delta_d)

        I[d] = I[d - step] * expd + psiu * S[d - step] + psi0 * S[d] + psid * S[d + step]
        L[d] = psi0

    d = end - step
    delta_u = (tau[d - step] - tau[d]) / mu
    expd = np.exp(-delta_u)
    if delta_u < 0.01:
        expd = 1.0 - delta_u + delta_u**2 / 2.0 - delta_u**3 / 6.0
        psi0 = delta_u / 2.0 - delta_u**2 / 6.0 + delta_u**3 / 24.0
        psiu = delta_u / 2.0 - delta_u**2 / 3.0 + delta_u**3 / 8.0
    else:
        psi0 = 1.0 - (1.0 - expd) / delta_u
        psiu = -expd + (1.0 - expd) / delta_u

    I[d] = I[d - step] * expd + psiu * S[d - step] + psi0 * S[d]
    L[d] = psi0
    return I, L


@jit(nopython=True)
def _short_characteristics_field(source, tau, mu, top, bottom):
    _, ncomp, nnu, nmu, nphi = source.shape
    stokes = np.zeros(source.shape)
    for f in range(nnu):
        tau_f = tau[:, f].copy()
        for m in range(nmu):
            for c in range(ncomp):
                for p in range(nphi):
                    boundary = bottom[c, f, m, p] if mu[m] > 0 else top[c, f, m, p]
                    ray, _ = short_characteristics(
                        tau_f, source[:, c, f, m, p].copy(), mu[m], boundary
                    )
                    stokes[:, c, f, m, p] = ray
    return stokes


def _checked_transfer_inputs(source, tau, mu, top, bottom):
    source, tau, mu, top, bottom = map(np.asarray, (source, tau, mu, top, bottom))
    if source.ndim != 5 or source.shape[1] != 3 or any(n == 0 for n in source.shape):
        raise ValueError("source must have non-empty shape [depth, 3, frequency, mu, phi]")
    ndepth, _, nnu, nmu, nphi = source.shape
    if tau.shape != (ndepth, nnu):
        raise ValueError(f"tau must have shape ({ndepth}, {nnu})")
    if mu.shape != (nmu,):
        raise ValueError(f"mu must have shape ({nmu},)")
    boundary_shape = (3, nnu, nmu, nphi)
    if top.shape != boundary_shape or bottom.shape != boundary_shape:
        raise ValueError(f"top and bottom must have shape {boundary_shape}")
    for name, value in (("source", source), ("tau", tau), ("mu", mu),
                        ("top", top), ("bottom", bottom)):
        if not np.isrealobj(value) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be real and finite")
    source, tau, mu, top, bottom = (
        np.asarray(value, dtype=float) for value in (source, tau, mu, top, bottom)
    )
    if np.any(tau < 0) or np.any(np.diff(tau, axis=0) < 0):
        raise ValueError("tau must be non-negative and non-decreasing with depth")
    if np.any(mu == 0) or np.any(np.abs(mu) > 1):
        raise ValueError("mu must be nonzero and between -1 and 1")
    return source, tau, mu, top, bottom


def solve_short_characteristics(source, tau, mu, top, bottom):
    """Parabolic short-characteristics solution of the I/Q/U transfer equations.

    Same arguments, units and boundary conventions as solve_unpolarized_transfer,
    but the source is interpolated parabolically between depth nodes. At least
    two depth points and strictly increasing tau are required. Returns
    [depth, 3, frequency, mu, phi].
    """
    source, tau, mu, top, bottom = _checked_transfer_inputs(source, tau, mu, top, bottom)
    if source.shape[0] < 2:
        raise ValueError("short characteristics needs at least two depth points")
    if np.any(np.diff(tau, axis=0) <= 0):
        raise ValueError("tau must strictly increase with depth")
    return _short_characteristics_field(source, tau, mu, top, bottom)


def solve_unpolarized_transfer(source, tau, mu, top, bottom):
    """Formal I/Q/U solution for fixed scalar opacity and linear cell sources.

    source is [depth, 3, frequency, mu, phi], tau [depth, frequency],
    and top and bottom [3, frequency, mu, phi]. Optical depth increases
    inward and mu > 0 is outward. Boundary values are used only for rays
    entering at the corresponding boundary. Returns the I/Q/U field.
    """
    source, tau, mu, top, bottom = _checked_transfer_inputs(source, tau, mu, top, bottom)
    ndepth = source.shape[0]
    stokes = np.empty(source.shape, dtype=float)
    for m, ray_mu in enumerate(mu):
        step = -1 if ray_mu > 0 else 1
        start = ndepth - 1 if ray_mu > 0 else 0
        stokes[start, :, :, m, :] = (bottom if ray_mu > 0 else top)[:, :, m, :]
        for k in range(start + step, ndepth if step > 0 else -1, step):
            previous = k - step
            with np.errstate(over="ignore"):
                dt = np.abs(tau[k] - tau[previous]) / abs(ray_mu)
            attenuation = np.exp(-dt)
            absorbed = -np.expm1(-dt)
            small = dt < 1e-2
            incoming_weight = np.empty_like(dt)
            t = dt[small]
            incoming_weight[small] = t * (
                0.5 + t * (-1/3 + t * (1/8 + t * (-1/30 + t * (1/144 - t/840))))
            )
            incoming_weight[~small] = absorbed[~small] / dt[~small] - attenuation[~small]
            outgoing_weight = absorbed - incoming_weight
            stokes[k, :, :, m, :] = (
                stokes[previous, :, :, m, :] * attenuation[None, :, None]
                + source[previous, :, :, m, :] * incoming_weight[None, :, None]
                + source[k, :, :, m, :] * outgoing_weight[None, :, None]
            )
    return stokes


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
