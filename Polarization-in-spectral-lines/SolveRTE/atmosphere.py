"""Data layer for a 1D plane-parallel grid solver.

Conventions
-----------
* Index 0 is the outermost (top) point and index N-1 the innermost, so the
  optical depth tau increases with the index and z decreases with it.
* mu = cos(theta) is measured from the outward vertical. mu > 0 is a ray
  travelling outward (towards the observer at the top), mu < 0 inward.
* B is given in the vertical frame as (B, theta_B, chi_B).
* J^K_Q and rho^K_Q arrays are indexed [depth, K, Q + 2] with K = 0, 1, 2 and
  Q = -2..2 (entries with |Q| > K stay zero). This matches idx(Q) = Q + 2 in
  Radiation_fun.py.
* Stokes order is (I, Q, U, V). The propagation matrix is stored as
  (eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V).
"""
from dataclasses import dataclass, fields
import numpy as np

K_B = 1.380649e-16      # erg/K
AMU = 1.66053907e-24    # g
C_LIGHT = 2.99792458e10  # cm/s


@dataclass
class Atmosphere1D:
    """Per-depth model. Every array has length N."""
    z: np.ndarray
    T: np.ndarray
    n_l: np.ndarray
    B: np.ndarray
    theta_B: np.ndarray
    chi_B: np.ndarray
    v_los: np.ndarray
    D2: np.ndarray        # depolarizing collision rate of the upper level [1/s]
    C_ul: np.ndarray      # inelastic de-excitation rate [1/s]
    vmic: np.ndarray = None

    def __post_init__(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                setattr(self, f.name, np.asarray(val, dtype=float))
        if self.vmic is None:
            self.vmic = np.zeros_like(self.z)
        n = len(self.z)
        for f in fields(self):
            if getattr(self, f.name).shape != (n,):
                raise ValueError(
                    f"{f.name} must have shape ({n},), got {getattr(self, f.name).shape}"
                )
        if n > 1 and not np.all(np.diff(self.z) < 0):
            raise ValueError("z must strictly decrease with index (index 0 is the top)")

    @property
    def N(self):
        return len(self.z)

    def point(self, k):
        """Scalars at depth index k."""
        return {f.name: getattr(self, f.name)[k] for f in fields(self) if f.name != "z"}


@dataclass
class LineModel:
    """Constant atomic parameters of a two-level line."""
    nu0: float        # line centre frequency [Hz]
    A_ul: float       # Einstein A [1/s]
    gJu: float
    gJl: float
    Ju: float
    Jl: float
    mass_amu: float

    def doppler_width(self, T, vmic=0.0):
        """Doppler width in frequency [Hz]."""
        v2 = 2 * K_B * np.asarray(T) / (self.mass_amu * AMU) + np.asarray(vmic) ** 2
        return self.nu0 * np.sqrt(v2) / C_LIGHT


@dataclass
class Grid:
    """Atmosphere, line, quadratures and solver state."""
    atm: Atmosphere1D
    line: LineModel
    mu: np.ndarray       # nodes in [-1, 1]
    w_mu: np.ndarray
    phi: np.ndarray      # azimuths in [0, 2 pi)
    w_phi: np.ndarray
    nu: np.ndarray       # offsets from line centre [Hz]
    w_nu: np.ndarray

    @classmethod
    def build(cls, atm, line, n_mu_half=4, n_phi=8, n_nu=61, nu_max_doppler=6.0):
        """Gauss-Legendre in mu on each hemisphere, uniform in phi and nu."""
        x, w = np.polynomial.legendre.leggauss(n_mu_half)
        x = 0.5 * (x + 1.0)           # map to (0, 1)
        w = 0.5 * w
        mu = np.concatenate([-x[::-1], x])
        w_mu = np.concatenate([w[::-1], w])

        phi = 2 * np.pi * np.arange(n_phi) / n_phi
        w_phi = np.full(n_phi, 2 * np.pi / n_phi)

        dnu = line.doppler_width(atm.T.max(), atm.vmic.max())
        nu = np.linspace(-nu_max_doppler, nu_max_doppler, n_nu) * dnu
        w_nu = np.full(n_nu, nu[1] - nu[0])
        return cls(atm, line, mu, w_mu, phi, w_phi, nu, w_nu)

    def allocate(self):
        """Create the arrays the iteration updates in place."""
        N, nmu, nphi, nnu = self.atm.N, len(self.mu), len(self.phi), len(self.nu)
        self.tau = np.zeros(N)
        self.JKQ = np.zeros((N, 3, 5), dtype=complex)
        self.rho = np.zeros((N, 3, 5), dtype=complex)
        self.eps = np.zeros((N, 4, nnu, nmu, nphi))
        self.eta = np.zeros((N, 7, nnu, nmu, nphi))
        self.stokes = np.zeros((N, 4, nnu, nmu, nphi))
        return self
