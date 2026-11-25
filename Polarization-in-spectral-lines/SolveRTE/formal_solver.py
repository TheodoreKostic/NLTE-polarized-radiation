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


@jit(nopython=True)
def _delo_weights(delta):
    """exp(-delta) and the linear-source weights of the upstream and downstream node.

    w_u = (1 - exp(-delta))/delta - exp(-delta), w_d = 1 - (1 - exp(-delta))/delta.
    Taylor series below delta = 1e-2 avoid cancellation.
    """
    expd = np.exp(-delta)
    if delta < 1e-2:
        w_u = delta * (0.5 + delta * (-1.0 / 3 + delta * (1.0 / 8 + delta * (
            -1.0 / 30 + delta * (1.0 / 144 + delta * (-1.0 / 840 + delta / 5760))))))
        w_d = delta * (0.5 + delta * (-1.0 / 6 + delta * (1.0 / 24 + delta * (
            -1.0 / 120 + delta * (1.0 / 720 + delta * (-1.0 / 5040 + delta / 40320))))))
    else:
        ratio = -np.expm1(-delta) / delta
        w_u = ratio - expd
        w_d = 1.0 - ratio
    return expd, w_u, w_d


@jit(nopython=True)
def _kprime(eta_row, out):
    """Fill out [4, 4] with K / eta_I - 1 (zero diagonal) from (eta_IQUV, rho_QUV)."""
    inv = 1.0 / eta_row[0]
    eq, eu, ev = eta_row[1] * inv, eta_row[2] * inv, eta_row[3] * inv
    rq, ru, rv = eta_row[4] * inv, eta_row[5] * inv, eta_row[6] * inv
    out[:, :] = 0.0
    out[0, 1], out[0, 2], out[0, 3] = eq, eu, ev
    out[1, 0], out[1, 2], out[1, 3] = eq, rv, -ru
    out[2, 0], out[2, 1], out[2, 3] = eu, -rv, rq
    out[3, 0], out[3, 1], out[3, 2] = ev, ru, -rq


@jit(nopython=True)
def _solve4(a, b):
    """Solve a x = b for a [4, 4] by Gaussian elimination with partial pivoting; a and b are overwritten."""
    for col in range(4):
        pivot = col
        for row in range(col + 1, 4):
            if abs(a[row, col]) > abs(a[pivot, col]):
                pivot = row
        if a[pivot, col] == 0.0:
            raise ValueError("singular DELO system")
        if pivot != col:
            for k in range(4):
                a[col, k], a[pivot, k] = a[pivot, k], a[col, k]
            b[col], b[pivot] = b[pivot], b[col]
        for row in range(col + 1, 4):
            factor = a[row, col] / a[col, col]
            for k in range(col, 4):
                a[row, k] -= factor * a[col, k]
            b[row] -= factor * b[col]
    x = np.zeros(4)
    for row in range(3, -1, -1):
        total = b[row]
        for k in range(row + 1, 4):
            total -= a[row, k] * x[k]
        x[row] = total / a[row, row]
    return x


@jit(nopython=True)
def delo_polarized_ray(eta, emis, dz, mu, I_boundary):
    """DELO-linear solution of the polarized transfer equation along one ray.

    Solves dI/ds = emis - K I with K built from eta = (eta_I, eta_Q, eta_U, eta_V,
    rho_Q, rho_U, rho_V) as in propagation_matrix. With dt = eta_I ds, K' = K/eta_I - 1
    and S = emis/eta_I the equation is dI/dt = -I + (S - K' I). The bracket is
    interpolated linearly in t across each layer, which gives, for upstream node u
    and downstream node d,
        (1 + w_d K'_d) I_d = (exp(-delta) - w_u K'_u) I_u + w_u S_u + w_d S_d,
    with delta = (eta_I,u + eta_I,d) / 2 * |dz| / |mu| and the weights of _delo_weights.
    K' = 0 reproduces solve_unpolarized_transfer for the same optical depths.

    Shapes: eta [depth, 7] (cm^-1), emis [depth, 4] (same units as eta times I),
    dz [depth - 1] with dz[k] = z[k] - z[k+1] > 0 (cm, index 0 is the top),
    mu a nonzero cosine, I_boundary [4] the incoming Stokes vector. Rays with
    mu > 0 travel upward from the bottom, mu < 0 downward from the top.
    Returns the Stokes vector at every depth, [depth, 4].
    """
    nd = emis.shape[0]
    if nd < 2 or mu == 0.0 or abs(mu) > 1.0:
        raise ValueError("need at least two depth points and a nonzero mu with |mu| <= 1")
    begin, step, end = (nd - 1, -1, -1) if mu > 0 else (0, 1, nd)
    for d in range(nd):
        if not eta[d, 0] > 0.0:
            raise ValueError("eta_I must be positive")
    for k in range(nd - 1):
        if not dz[k] > 0.0:
            raise ValueError("dz must be positive")

    stokes = np.zeros((nd, 4))
    stokes[begin, :] = I_boundary
    kp_u = np.zeros((4, 4))
    kp_d = np.zeros((4, 4))
    _kprime(eta[begin], kp_u)
    system = np.zeros((4, 4))
    rhs = np.zeros(4)
    for d in range(begin + step, end, step):
        u = d - step
        _kprime(eta[d], kp_d)
        delta = 0.5 * (eta[u, 0] + eta[d, 0]) * dz[min(u, d)] / abs(mu)
        expd, w_u, w_d = _delo_weights(delta)
        for i in range(4):
            rhs[i] = (expd * stokes[u, i] + w_u * emis[u, i] / eta[u, 0]
                      + w_d * emis[d, i] / eta[d, 0])
            for j in range(4):
                rhs[i] -= w_u * kp_u[i, j] * stokes[u, j]
                system[i, j] = w_d * kp_d[i, j] + (1.0 if i == j else 0.0)
        stokes[d, :] = _solve4(system, rhs)
        kp_u[:, :] = kp_d
    return stokes


@jit(nopython=True)
def _delo_field(eta, emis, dz, mu, top, bottom):
    """Run delo_polarized_ray over every wavelength, mu and phi.

    eta [NL, Nmu, Nphi, ND, 7], emis [NL, Nmu, Nphi, ND, 4], top and bottom
    [NL, Nmu, Nphi, 4]; returns [NL, Nmu, Nphi, ND, 4].
    """
    nnu, nmu, nphi, nd = eta.shape[0], eta.shape[1], eta.shape[2], eta.shape[3]
    result = np.zeros((nnu, nmu, nphi, nd, 4))
    for f in range(nnu):
        for m in range(nmu):
            for p in range(nphi):
                boundary = bottom[f, m, p] if mu[m] > 0 else top[f, m, p]
                result[f, m, p] = delo_polarized_ray(
                    eta[f, m, p], emis[f, m, p], dz, mu[m], boundary)
    return result


def solve_delo(emis, eta, z, mu, top, bottom):
    """Polarized DELO-linear formal solution of I, Q, U, V for every ray.

    emis [depth, 4, frequency, mu, phi] and eta [depth, 7, frequency, mu, phi] are the
    arrays filled by grid_opacity.fill_grid (grid.emis, grid.eta), with eta ordered
    (eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V) in cm^-1. z [depth] is the height
    in cm, strictly decreasing (index 0 is the top). mu [mu] holds nonzero cosines;
    mu > 0 rays travel upward from bottom, mu < 0 rays downward from top. top and
    bottom [4, frequency, mu, phi] are the incoming Stokes vectors; only the boundary
    a ray enters from is used. Returns the Stokes vector [depth, 4, frequency, mu, phi],
    whose depth index 0 is the emergent radiation of the mu > 0 rays.
    """
    emis, eta, z, mu, top, bottom = (
        np.asarray(a) for a in (emis, eta, z, mu, top, bottom))
    for name, value in (("emis", emis), ("eta", eta), ("z", z), ("mu", mu),
                        ("top", top), ("bottom", bottom)):
        if not np.isrealobj(value) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be real and finite")
    if emis.ndim != 5 or emis.shape[0] < 2 or emis.shape[1] != 4:
        raise ValueError("emis must have shape [depth >= 2, 4, frequency, mu, phi]")
    depth, _, nnu, nmu, nphi = emis.shape
    if eta.shape != (depth, 7, nnu, nmu, nphi):
        raise ValueError(f"eta must have shape {(depth, 7, nnu, nmu, nphi)}")
    if z.shape != (depth,) or np.any(np.diff(z) >= 0):
        raise ValueError(f"z must have shape ({depth},) and strictly decrease with depth index")
    if mu.shape != (nmu,) or np.any(mu == 0) or np.any(np.abs(mu) > 1):
        raise ValueError(f"mu must have shape ({nmu},), be nonzero and satisfy |mu| <= 1")
    if top.shape != (4, nnu, nmu, nphi) or bottom.shape != top.shape:
        raise ValueError(f"top and bottom must have shape {(4, nnu, nmu, nphi)}")
    if np.any(eta[:, 0] <= 0):
        raise ValueError("eta_I must be positive")

    def rays_first(a, depth_axes):
        return np.ascontiguousarray(a.transpose(depth_axes), dtype=float)

    result = _delo_field(
        rays_first(eta, (2, 3, 4, 0, 1)), rays_first(emis, (2, 3, 4, 0, 1)),
        np.ascontiguousarray(-np.diff(z), dtype=float), np.asarray(mu, dtype=float),
        rays_first(top, (1, 2, 3, 0)), rays_first(bottom, (1, 2, 3, 0)),
    )
    return np.ascontiguousarray(result.transpose(3, 4, 0, 1, 2))
