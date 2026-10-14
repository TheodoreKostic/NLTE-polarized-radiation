"""Tensor machinery for the grid: J^K_Q from Stokes, Hanle/collisions, emissivity.

Builds on the existing T (Radiation_fun.py) and wigner_D2 (functions_prt.py).
Everything is expressed in the vertical (local) reference frame, where the ray
direction is (theta, chi) = (arccos(mu), phi) and gamma is the angle of the
Stokes-Q reference axis (gamma = 0 means Q is along the projection of the
vertical onto the plane of the sky, LL04 Table 5.6).

Not included yet: Doppler shifts, rank-1 (orientation) density-matrix elements,
lower-level polarization, magneto-optical terms.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from functions_prt import wigner_D2  # noqa: E402
from Radiation_fun import T  # noqa: E402
from Hanle_fun import hanle_parameter  # noqa: E402

QS = np.arange(-2, 3)


def tensor_basis(mu, phi, gamma=0.0):
    """T^K_Q(i, Omega) on the angular grid.

    Returns array [i (I,Q,U,V), K (0..2), Q+2, nmu, nphi].
    """
    theta = np.arccos(np.clip(mu, -1.0, 1.0))[:, None] * np.ones((1, len(phi)))
    chi = np.ones((len(mu), 1)) * np.asarray(phi)[None, :]
    return tensor_basis_angles(theta, chi, gamma)


def tensor_basis_angles(theta, chi, gamma=0.0):
    """T^K_Q(i) for arbitrary-shaped angle arrays; returns [4, 3, 5, *theta.shape]."""
    theta = np.asarray(theta, dtype=float)
    chi = np.broadcast_to(np.asarray(chi, dtype=float), theta.shape)
    gam = np.broadcast_to(np.asarray(gamma, dtype=float), theta.shape)
    basis = np.zeros((4, 3, 5) + theta.shape, dtype=complex)
    for i in range(4):
        for K in range(3):
            for Q in QS:
                if abs(Q) <= K:
                    basis[i, K, Q + 2] = T(i, K, int(Q), theta, chi, gam)
    # Radiation_fun uses sqrt(3)/2 for T^1_0(V); LL04 Table 5.6 gives sqrt(3/2) cos(theta)
    basis[3, 1, 2] = np.sqrt(1.5) * np.cos(theta)
    # sign fixed so that weak-field V = +nu_L B cos(theta) dI/dnu (LL04 Eq. 9.x, B towards observer)
    basis[3, 1] *= -1.0
    return basis


def gaussian_profile(grid):
    """Normalised Gaussian profile phi(k, nu) from the local Doppler width."""
    dnu = grid.line.doppler_width(grid.atm.T, grid.atm.vmic)
    prof = np.exp(-(grid.nu[None, :] / dnu[:, None]) ** 2)
    return prof / (prof @ grid.w_nu)[:, None]


def jkq_from_stokes(grid, stokes, basis=None, profile=None):
    """J^K_Q(k) = (1/4pi) sum_{nu,mu,phi} w phi_nu(k) sum_i T^K_Q(i) S_i.

    stokes: [N, 4, nnu, nmu, nphi]. Returns [N, 3, 5].
    """
    if basis is None:
        basis = tensor_basis(grid.mu, grid.phi)
    if profile is None:
        profile = gaussian_profile(grid)
    w_ang = grid.w_mu[:, None] * grid.w_phi[None, :]
    # angle-and-frequency weighted Stokes: [N, 4, nmu, nphi]
    S = np.einsum("kinmp,kn,n,mp->kimp", stokes, profile, grid.w_nu, w_ang)
    return np.einsum("iKQmp,kimp->kKQ", basis, S) / (4 * np.pi)


def rotate_tensor(rho_K, alpha, beta, gamma):
    """Rotate a rank-2 tensor array [5] with the Wigner matrix D^2(alpha, beta, gamma)."""
    return wigner_D2(alpha, beta, gamma) @ rho_K


def rotate_stokes(S, gamma):
    """Rotate the Stokes reference axis by gamma. S[..., 0:4] = (I, Q, U, V)."""
    c, s = np.cos(2 * gamma), np.sin(2 * gamma)
    out = np.array(S, dtype=float, copy=True)
    out[..., 1] = c * S[..., 1] + s * S[..., 2]
    out[..., 2] = -s * S[..., 1] + c * S[..., 2]
    return out


def hanle_operator_general(Hu, delta2, theta_B, chi_B):
    """D diag(1 / (1 + delta2 + i Q Hu)) D^dagger for rank 2.

    Same operator as Hanle_fun.hanle_operator, with depolarizing collisions
    delta2 = D^(2) / A_ul. In the magnetic frame the Hanle term is diagonal.
    """
    diag = np.diag(1.0 / (1.0 + delta2 + 1j * QS * Hu))
    D = wigner_D2(chi_B, theta_B, 0.0)
    return D @ diag @ D.conj().T


def solve_rho_point(JKQ, line, B, theta_B, chi_B, D2, C_ul=0.0, w2=1.0):
    """Upper-level rho^K_Q at one point for a two-level atom, no lower-level polarization.

    rho^0_0 = J^0_0 / (1 + C_ul/A) is a placeholder for the full statistical
    equilibrium and equals J^0_0 for C_ul = 0 (as in hanle_polarization_corrected).
    rho^2_Q = w2 * [D H D^dagger J^2]_Q with the rank-2 relaxation
    1 + (C_ul + D2)/A + i Q Hu. w2 is the polarizability factor (1 for Ju=1, Jl=0).
    """
    rho = np.zeros((3, 5), dtype=complex)
    rho[0, 2] = JKQ[0, 2] / (1.0 + C_ul / line.A_ul)
    Hu = hanle_parameter(B, line.gJu, line.A_ul)
    H = hanle_operator_general(Hu, (D2 + C_ul) / line.A_ul, theta_B, chi_B)
    rho[2] = w2 * (H @ JKQ[2])
    return rho


def solve_rho(grid, w2=1.0):
    """Apply solve_rho_point at every depth, using grid.JKQ. Stores and returns grid.rho."""
    a = grid.atm
    for k in range(a.N):
        grid.rho[k] = solve_rho_point(
            grid.JKQ[k], grid.line, a.B[k], a.theta_B[k], a.chi_B[k], a.D2[k], a.C_ul[k], w2
        )
    return grid.rho


def emissivity_from_rho_grid(rho, basis):
    """eps_i(Omega) = sum_{K,Q} (-1)^Q T^K_Q(i, Omega) conj(rho^K_{-Q}).

    The conjugate follows the convention of Chapter_13_magnetic_branch_plots.py
    (used by Special_cases.py). rho: [3, 5] at one point (or [N, 3, 5]); basis
    from tensor_basis. Returns [4, nmu, nphi] (or [N, 4, nmu, nphi]); real part.
    """
    phase = (-1.0) ** QS
    rho_m = np.conj(rho[..., ::-1])  # conj(rho^K_{-Q})
    eps = np.einsum("iKQmp,...KQ->...imp", basis, phase * rho_m)
    return eps.real
