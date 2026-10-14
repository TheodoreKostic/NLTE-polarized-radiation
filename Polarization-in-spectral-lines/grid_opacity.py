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
from scipy.special import wofz

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from functions_prt import wigner_D2  # noqa: E402
from Profile_fun import W3  # noqa: E402
from atmosphere import C_LIGHT, K_B  # noqa: E402
from grid_tensors import QS, tensor_basis_angles  # noqa: E402

H_PLANCK = 6.62607015e-27
NU_LARMOR = 1.3996e6  # Hz/G, same value as Profile_fun / Special_cases
POLE_TOL = 1e-10


@lru_cache(maxsize=1)
def _phi_coefficients():
    """C[K, K', Q+2, Mu+1, Mu'+1]: angular coefficients of Phi^Q_{KK'} for Jl=0, Ju=1."""
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
    """Vectorised Phi^Q_{KK'}(x) with the Voigt-Faraday profile; shape [3, 3, 5, nx].

    Same definition as Profile_fun.Phi_generalized (x in Doppler units).
    """
    x = np.asarray(x, dtype=float)
    # phi_M(x) = wofz(x - M vH + i a) / sqrt(pi)
    prof = np.array([wofz(x - M * vH + 1j * a) for M in (-1, 0, 1)]) / np.sqrt(np.pi)
    pair = 0.5 * (prof[:, None, :] + np.conj(prof)[None, :, :])  # [Mu, Mu', nx]
    return np.einsum("klqmn,mnx->klqx", _phi_coefficients(), pair)


def phi_table_disp(x, vH, a):
    """Dispersion counterpart (Faraday-Voigt psi profiles) of phi_table; shape [3, 3, 5, nx]."""
    x = np.asarray(x, dtype=float)
    prof = np.array([wofz(x - M * vH + 1j * a) for M in (-1, 0, 1)]) / np.sqrt(np.pi)
    pair = (prof[:, None, :] - np.conj(prof)[None, :, :]) / (2j)
    return np.einsum("klqmn,mnx->klqx", _phi_coefficients(), pair).real


def phi00_norm():
    """c00 such that Phi^0_{00}(x) = c00 * phi(x) at B = 0."""
    return _phi_coefficients()[0, 0, 2].sum()


def magnetic_frame_geometry(mu, phi, theta_B, chi_B, gamma):
    """Observer angles in the magnetic frame and Q/U back-rotation, for a grid of rays.

    Replicates prepare_magnetic_branch_state (B_vector=None,
    mode "fixed_gamma_rotate_qu_back"). Returns theta_mag, chi_mag, back_rot
    with shape [nmu, nphi].
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
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    return q * c2 + u * s2, -q * s2 + u * c2


def planck(nu, T):
    return 2 * H_PLANCK * nu**3 / C_LIGHT**2 / np.expm1(H_PLANCK * nu / (K_B * T))


def eta0(line, n_l):
    """Frequency-integrated line absorption coefficient [cm^-1 Hz] (no stimulated emission)."""
    lam = C_LIGHT / line.nu0
    gratio = (2 * line.Ju + 1) / (2 * line.Jl + 1)
    return gratio * lam**2 * line.A_ul * n_l / (8 * np.pi)


def rho2_magnetic(J2, line, B, theta_B, chi_B, D2, C_ul):
    """Upper-level rho^2_Q in the magnetic frame (the form used by prepare_magnetic_branch_state)."""
    Hu = 2 * np.pi * NU_LARMOR * B * line.gJu / line.A_ul
    jmag = wigner_D2(chi_B, theta_B, 0.0).conj().T @ J2
    return jmag / (1.0 + (D2 + C_ul) / line.A_ul + 1j * QS * Hu)


def point_emission_and_propagation(
    JKQ, line, mu, phi, nu, T, n_l, B, theta_B, chi_B, v_los, vmic, D2, C_ul,
    gamma=0.0, rho_sign=1.0,
):
    """eps [4, nnu, nmu, nphi] and eta [7, nnu, nmu, nphi] at one depth point.

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
        x = (nu - line.nu0 * v_los * m / C_LIGHT) / dnu_D
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


def fill_grid(grid, gamma=0.0, rho_sign=1.0):
    """Fill grid.eps and grid.eta at every depth point from grid.JKQ and the atmosphere."""
    a = grid.atm
    for k in range(a.N):
        grid.eps[k], grid.eta[k] = point_emission_and_propagation(
            grid.JKQ[k], grid.line, grid.mu, grid.phi, grid.nu,
            a.T[k], a.n_l[k], a.B[k], a.theta_B[k], a.chi_B[k], a.v_los[k], a.vmic[k],
            a.D2[k], a.C_ul[k], gamma=gamma, rho_sign=rho_sign,
        )
    return grid
