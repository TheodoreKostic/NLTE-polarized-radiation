"""Frequency-resolved emissivity and propagation matrix at every grid point.

Two-level atom Jl = 0 -> Ju = 1, unpolarized lower level, complete
redistribution, same profile functions Phi^Q_{KK'} (LL04 Eq. 10.40) and same
magnetic-frame geometry as Chapter_13_magnetic_branch_plots.py /
Special_cases.py:

    eps_i(nu, Omega) = eta0/(c00 dnu_D) * Re[ beta0 * sum_K' Phi^0_{0K'} T^K'_0(i) J00
                         + sum_{K',Q} (-1)^Q Phi^Q_{2K'} T^K'_Q(i) conj(rho^2_{-Q}) ]
                       + (1 - beta0) * B_nu(T) * eta_i(nu, Omega)
    eta_i = eta0/(c00 dnu_D) * Re[sum_K' Phi^0_{0K'} T^K'_0(i)]        (i = I,Q,U,V)
    rho_i = sign * eta0/(c00 dnu_D) * Im[sum_K' Phi^0_{0K'} T^K'_0(i)]  (i = Q,U,V)

with T^K'_Q evaluated in the magnetic frame, and Q/U (and rho_Q/rho_U) rotated
back to the reference of the vertical frame, as in the "fixed_gamma_rotate_qu_back"
mode. rho^2_Q (magnetic frame) = [D^dagger J^2]_Q / (1 + (C_ul + D2)/A + i Q Hu).

Units: eta in cm^-1, eps in erg s^-1 cm^-3 Hz^-1 sr^-1 when J^K_Q is in
erg s^-1 cm^-2 Hz^-1 sr^-1 (J^K_Q is the profile-weighted line average).
"""
import os
import sys
from functools import lru_cache

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PARENT)
for _p in (_HERE, _PARENT, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from Old.functions_prt import wigner_D2  # noqa: E402
from Profile_fun import W3  # noqa: E402
from atmosphere import C_LIGHT, K_B  # noqa: E402
from grid_tensors import QS, tensor_basis_angles  # noqa: E402
from line_profiles import NU_LARMOR, voigt_function  # noqa: E402

H_PLANCK = 6.62607015e-27
POLE_TOL = 1e-10


@lru_cache(maxsize=1)
def _phi_coefficients():
    """Angular coefficients of the generalized profiles Phi^Q_{KK'} for Jl=0, Ju=1.

    Returns C[K, K', Q+2, Mu+1, Mu'+1] built from Wigner 3-j symbols (LL04 Eq. 10.40).
    Phi^Q_{KK'}(x) = sum_{Mu,Mu'} C[...] * (profile pair of Zeeman components Mu, Mu').
    Independent of frequency and field, so it is computed once and cached.
    """
    Ju, Jl, Ml = 1, 0, 0
    C = np.zeros((3, 3, 5, 3, 3))
    for K in range(3):
        for Kp in range(3):
            pref = np.sqrt(3 * (2 * Ju + 1) * (2 * K + 1) * (2 * Kp + 1))
            for Q in QS:
                for Mu in (-1, 0, 1):
                    for Mup in (-1, 0, 1):
                        q, qp = Mu, Mup
                        C[K, Kp, Q + 2, Mu + 1, Mup + 1] = pref * (
                            (-1) ** (1 + Ju - Mu + qp)
                            * W3(Ju, Jl, 1, -Mu, Ml, q)
                            * W3(Ju, Jl, 1, -Mup, Ml, qp)
                            * W3(Ju, Ju, K, Mu, -Mup, -int(Q))
                            * W3(1, 1, Kp, q, -qp, -int(Q))
                        )
    return C


def phi_table(x, vH, a):
    """Vectorised absorption-type profiles Phi^Q_{KK'}(x), shape [3, 3, 5, nx].

    x is the frequency in Doppler units (positive on the red-to-line-center side as
    defined in point_emission_and_propagation), vH = gJu * nu_L * B / dnu_D is the
    Zeeman shift in Doppler units and a the Voigt damping parameter. Each of the three
    Zeeman components (M = -1, 0, 1) is a Voigt profile shifted by M*vH; the pair
    products are projected on the angular coefficients of _phi_coefficients.
    Same definition as Profile_fun.Phi_generalized.
    """
    x = np.asarray(x, dtype=float)
    # phi_M(x) = wofz(x - M vH + i a) / sqrt(pi)
    prof = np.array([voigt_function(x - M * vH, a) for M in (-1, 0, 1)])
    pair = 0.5 * (prof[:, None, :] + np.conj(prof)[None, :, :])  # [Mu, Mu', nx]
    return np.einsum("klqmn,mnx->klqx", _phi_coefficients(), pair)


def phi_table_disp(x, vH, a):
    """Dispersion counterpart of phi_table, shape [3, 3, 5, nx].

    Built from the imaginary parts (Faraday-Voigt functions) of the Zeeman component
    profiles. Used for the magneto-optical terms rho_Q, rho_U, rho_V of the
    propagation matrix; same arguments as phi_table.
    """
    x = np.asarray(x, dtype=float)
    prof = np.array([voigt_function(x - M * vH, a) for M in (-1, 0, 1)])
    pair = (prof[:, None, :] - np.conj(prof)[None, :, :]) / (2j)
    return np.einsum("klqmn,mnx->klqx", _phi_coefficients(), pair).real


def phi00_norm():
    """Normalization c00 such that Phi^0_{00}(x) = c00 * phi(x) at B = 0.

    Dividing by it makes eta0/(c00*dnu_D) * Phi^0_00 integrate to eta0 over frequency.
    """
    return _phi_coefficients()[0, 0, 2].sum()


def magnetic_frame_geometry(mu, phi, theta_B, chi_B, gamma):
    """Observer geometry in the magnetic frame, for a grid of rays.

    The magnetic frame has its z axis along B. For each ray (mu, phi) of the vertical
    frame this returns the ray's polar/azimuthal angles in the magnetic frame
    (theta_mag, chi_mag) and the angle by which Q/U must be rotated to go back to the
    vertical-frame reference direction (back_rot). The Stokes reference axis is
    gamma from the theta direction. Rays at the magnetic pole have an undefined
    azimuth and get chi = 0. Replicates prepare_magnetic_branch_state (B_vector=None,
    mode "fixed_gamma_rotate_qu_back"). Shapes [nmu, nphi].
    """
    theta = np.arccos(np.clip(mu, -1.0, 1.0))[:, None] * np.ones((1, len(phi)))
    chi = np.ones((len(mu), 1)) * np.asarray(phi)[None, :]
    st, ct, sc, cc = np.sin(theta), np.cos(theta), np.sin(chi), np.cos(chi)
    los = np.stack([st * cc, st * sc, ct], axis=-1)
    e_th = np.stack([ct * cc, ct * sc, -st], axis=-1)
    e_ch = np.stack([-sc, cc, np.zeros_like(sc)], axis=-1)
    qref = np.cos(gamma) * e_th + np.sin(gamma) * e_ch

    cz, sz = np.cos(-chi_B), np.sin(-chi_B)
    cy, sy = np.cos(-theta_B), np.sin(-theta_B)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot = ry @ rz
    los_m = los @ rot.T
    qref_m = qref @ rot.T
    qref_m = qref_m - np.sum(qref_m * los_m, axis=-1, keepdims=True) * los_m
    qref_m /= np.linalg.norm(qref_m, axis=-1, keepdims=True)

    theta_m = np.arccos(np.clip(los_m[..., 2], -1.0, 1.0))
    at_pole = np.hypot(los_m[..., 0], los_m[..., 1]) <= POLE_TOL
    chi_m = np.where(at_pole, 0.0, np.arctan2(los_m[..., 1], los_m[..., 0]))
    theta_m = np.where(at_pole, np.where(los_m[..., 2] >= 0.0, 0.0, np.pi), theta_m)

    sm, cm, scm, ccm = np.sin(theta_m), np.cos(theta_m), np.sin(chi_m), np.cos(chi_m)
    e_th_m = np.stack([cm * ccm, cm * scm, -sm], axis=-1)
    e_ch_m = np.stack([-scm, ccm, np.zeros_like(scm)], axis=-1)
    gamma_transport = np.arctan2(
        np.sum(qref_m * e_ch_m, axis=-1), np.sum(qref_m * e_th_m, axis=-1)
    )
    return theta_m, chi_m, gamma_transport - gamma


def _rotate_pair(q, u, psi):
    """Rotate the (Q, U) pair by reference-axis angle psi: Q' = Q cos2psi + U sin2psi, U' = -Q sin2psi + U cos2psi."""
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    return q * c2 + u * s2, -q * s2 + u * c2


def planck(nu, T):
    """Planck function B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1]; used for the thermal (collisional) part of eps."""
    return 2 * H_PLANCK * nu**3 / C_LIGHT**2 / np.expm1(H_PLANCK * nu / (K_B * T))


def eta0(line, n_l):
    """Frequency-integrated line absorption coefficient [cm^-1 Hz] (no stimulated emission)."""
    lam = C_LIGHT / line.nu0
    gratio = (2 * line.Ju + 1) / (2 * line.Jl + 1)
    return gratio * lam**2 * line.A_ul * n_l / (8 * np.pi)


def rho2_magnetic(J2, line, B, theta_B, chi_B, D2, C_ul):
    """Upper-level rho^2_Q in the magnetic frame (the form used by prepare_magnetic_branch_state).

    The radiation tensor J^2 (vertical frame) is rotated into the frame where B is the
    quantization axis, where the Hanle operator is diagonal:
        rho^2_Q = [D^dagger J^2]_Q / (1 + (D2 + C_ul)/A + i Q Hu),   Hu = 2 pi nu_L gJu B / A.
    Collisions (elastic D2 and inelastic C_ul) damp every Q equally; the magnetic term
    i Q Hu rotates and damps Q != 0 components (Hanle effect). B = 0 gives a pure damping.
    """
    Hu = 2 * np.pi * NU_LARMOR * B * line.gJu / line.A_ul
    jmag = wigner_D2(chi_B, theta_B, 0.0).conj().T @ J2
    return jmag / (1.0 + (D2 + C_ul) / line.A_ul + 1j * QS * Hu)


def point_emission_and_propagation(
    JKQ, line, mu, phi, nu, T, n_l, B, theta_B, chi_B, v_los, vmic, D2, C_ul,
    gamma=0.0, rho_sign=1.0,
):
    """Local emissivity and propagation matrix at one depth point for ONE magnetic field.

    Inputs: JKQ [3, 5] radiation tensors (index [K, Q+2], vertical frame); mu, phi ray
    grid; nu offsets from line center; T, n_l the temperature and lower-level density;
    (B, theta_B, chi_B) the field vector; v_los, vmic line-of-sight and micro-turbulent
    velocities; D2, C_ul collisional rates; gamma Stokes reference angle.
    Steps: Doppler width and damping -> Zeeman profile tables Phi -> radiation-tensor
    basis in the magnetic frame -> rho^2 (Hanle + collisions) -> eps and eta per ray ->
    rotate Q/U back to the vertical-frame reference -> add the thermal part
    (1 - beta0) B_nu eta.
    eps [4, nnu, nmu, nphi] and eta [7, nnu, nmu, nphi] at one depth point.
    For an unresolved field on top of this field use
    point_emission_and_propagation_unresolved.

    eta order: (eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V).
    """
    dnu_D = line.doppler_width(T, vmic)
    a = line.A_ul / (4 * np.pi * dnu_D)
    vH = line.gJu * NU_LARMOR * B / dnu_D
    c00 = phi00_norm()
    amp = eta0(line, n_l) / (c00 * dnu_D)

    theta_m, chi_m, back = magnetic_frame_geometry(mu, phi, theta_B, chi_B, gamma)
    basis = tensor_basis_angles(theta_m, chi_m, gamma)  # [4, 3, 5, nmu, nphi]
    rho2 = rho2_magnetic(JKQ[2], line, B, theta_B, chi_B, D2, C_ul)
    beta0 = 1.0 / (1.0 + C_ul / line.A_ul)
    j00 = JKQ[0, 2]
    sign = (-1.0) ** QS

    nnu, nmu, nphi = len(nu), len(mu), len(phi)
    eps = np.zeros((4, nnu, nmu, nphi))
    eta = np.zeros((7, nnu, nmu, nphi))

    for im, m in enumerate(mu):
        # nu is the offset nu - nu0; Phi uses LL04's reduced frequency x = (nu0 - nu_shifted)/dnu_D
        x = (-nu + line.nu0 * v_los * m / C_LIGHT) / dnu_D
        tab = phi_table(x, vH, a)  # [3, 3, 5, nx]
        # absorption-like K=0 block: sum_K' Phi^0_{0K'}(x) T^K'_0(i); [4, nx, nphi]
        absorb = np.einsum("kx,ikp->ixp", tab[0, :, 2], basis[:, :, 2, im])
        # rank-2 block: sum_{K',Q} (-1)^Q Phi^Q_{2K'} T^K'_Q(i) conj(rho2_{-Q})
        scat2 = np.einsum(
            "kqx,ikqp,q->ixp", tab[2], basis[..., im, :], sign * np.conj(rho2[::-1])
        )
        tab_d = phi_table_disp(x, vH, a)
        disp = np.einsum("kx,ikp->ixp", tab_d[0, :, 2], basis[:, :, 2, im].real)
        eta_c = amp * absorb
        eps_c = amp * (beta0 * j00 * absorb + scat2)

        comp = np.concatenate([eta_c.real, rho_sign * amp * disp[1:]])
        eta[:, :, im, :] = comp
        # rotate Q/U and rho_Q/rho_U back to the vertical-frame reference
        for (cq, cu) in ((1, 2), (4, 5)):
            q2, u2 = _rotate_pair(eta[cq, :, im, :], eta[cu, :, im, :], back[im][None, :])
            eta[cq, :, im, :], eta[cu, :, im, :] = q2, u2

        s = eps_c.real
        eq, eu = _rotate_pair(s[1], s[2], back[im][None, :])
        eps[0, :, im, :], eps[1, :, im, :] = s[0], eq
        eps[2, :, im, :], eps[3, :, im, :] = eu, s[3]

    if beta0 < 1.0:
        eps += (1.0 - beta0) * planck(line.nu0, T) * eta[[0, 1, 2, 3]]
    return eps, eta


def isotropic_field_directions(n_theta=8, n_chi=8):
    """Quadrature nodes and weights for averaging over directions uniform on the sphere.

    Gauss-Legendre in cos(theta) (n_theta nodes) times a uniform midpoint rule in
    chi (n_chi nodes). Returns flat arrays (theta, chi, weights) of length
    n_theta * n_chi; weights sum to 1, so sum(w * f) is the sphere average of f.
    The rule is exact for spherical harmonics up to degree 2*n_theta - 1 in
    cos(theta) and n_chi - 1 in chi, enough for the rank-2 radiation tensors.
    """
    x, w = np.polynomial.legendre.leggauss(n_theta)
    chi = 2 * np.pi * (np.arange(n_chi) + 0.5) / n_chi
    th = np.repeat(np.arccos(x), n_chi)
    ch = np.tile(chi, n_theta)
    wt = np.repeat(w / 2.0, n_chi) / n_chi
    return th, ch, wt


def total_field_samples(B, theta_B, chi_B, B_unres, n_theta=8, n_chi=8):
    """Field vectors B_mean + B_unres * n for n uniform on the sphere.

    The mean field (B, theta_B, chi_B) is the resolved field; B_unres is the
    strength of the unresolved random component added on top of it. Returns
    (B_tot, theta_tot, chi_tot, weights): the strength and direction of each
    sampled total field and its quadrature weight (weights sum to 1).
    If the sum vanishes the direction is arbitrary and set to (0, 0); this is
    harmless because the Zeeman/Hanle terms are then zero.
    """
    th, ch, wt = isotropic_field_directions(n_theta, n_chi)
    mean = B * np.array([np.sin(theta_B) * np.cos(chi_B),
                         np.sin(theta_B) * np.sin(chi_B),
                         np.cos(theta_B)])
    rnd = B_unres * np.stack([np.sin(th) * np.cos(ch), np.sin(th) * np.sin(ch), np.cos(th)], -1)
    tot = mean[None, :] + rnd
    b_tot = np.linalg.norm(tot, axis=-1)
    safe = np.where(b_tot > 0, b_tot, 1.0)
    theta_tot = np.where(b_tot > 0, np.arccos(np.clip(tot[:, 2] / safe, -1.0, 1.0)), 0.0)
    chi_tot = np.where(b_tot > 0, np.arctan2(tot[:, 1], tot[:, 0]), 0.0)
    return b_tot, theta_tot, chi_tot, wt


def point_emission_and_propagation_unresolved(
    JKQ, line, mu, phi, nu, T, n_l, B, theta_B, chi_B, B_unres, v_los, vmic, D2, C_ul,
    gamma=0.0, rho_sign=1.0, n_theta=8, n_chi=8,
):
    """eps and eta at one depth point for a mean field plus an unresolved isotropic component.

    The physical field is B_mean + B_unres * n, with n a random direction uniform on
    the sphere (B_unres = 0 reduces exactly to point_emission_and_propagation; B = 0
    gives a purely isotropic field of strength B_unres). For each sampled total field
    the full local physics is evaluated (Zeeman profiles, Hanle-modified rho^2, Faraday
    terms) and eps and all seven eta/rho coefficients are averaged with the same
    weights. Averaging the coefficients, not the Stokes profiles, is what the transfer
    equation needs for a sub-resolution field; averaging eps and eta together keeps
    the emission and absorption mutually consistent.

    Accuracy: n_theta x n_chi direction nodes. With B_unres >> B or small Hu the default
    8x8 is accurate to ~1e-5 or better. When B ~ B_unres and Hu >> 1 the total field
    can pass near zero and the integrand is sharply peaked, so raise n_theta, n_chi
    (24x24 gives ~1e-8 at B = B_unres = 5 G, but ~3e-3 at 50 G) and check convergence.
    """
    b_tot, t_tot, c_tot, wt = total_field_samples(B, theta_B, chi_B, B_unres, n_theta, n_chi)
    eps = eta = 0.0
    for b, t, c, w in zip(b_tot, t_tot, c_tot, wt):
        e, h = point_emission_and_propagation(
            JKQ, line, mu, phi, nu, T, n_l, b, t, c, v_los, vmic, D2, C_ul,
            gamma=gamma, rho_sign=rho_sign,
        )
        eps = eps + w * e
        eta = eta + w * h
    return eps, eta


def fill_grid(grid, gamma=0.0, rho_sign=1.0, n_theta=8, n_chi=8):
    """Fill grid.eps and grid.eta at every depth point from grid.JKQ and the atmosphere.

    Depths with atm.B_unres > 0 use the mean-plus-unresolved-field average
    (point_emission_and_propagation_unresolved); all others use the single mean
    field directly. n_theta, n_chi set the direction quadrature of that average.
    """
    a = grid.atm
    for k in range(a.N):
        if a.B_unres[k] > 0:
            grid.eps[k], grid.eta[k] = point_emission_and_propagation_unresolved(
                grid.JKQ[k], grid.line, grid.mu, grid.phi, grid.nu,
                a.T[k], a.n_l[k], a.B[k], a.theta_B[k], a.chi_B[k], a.B_unres[k],
                a.v_los[k], a.vmic[k], a.D2[k], a.C_ul[k],
                gamma=gamma, rho_sign=rho_sign, n_theta=n_theta, n_chi=n_chi,
            )
            continue
        grid.eps[k], grid.eta[k] = point_emission_and_propagation(
            grid.JKQ[k], grid.line, grid.mu, grid.phi, grid.nu,
            a.T[k], a.n_l[k], a.B[k], a.theta_B[k], a.chi_B[k], a.v_los[k], a.vmic[k],
            a.D2[k], a.C_ul[k], gamma=gamma, rho_sign=rho_sign,
        )
    return grid
