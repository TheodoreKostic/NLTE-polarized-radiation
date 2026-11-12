"""Atmosphere, atomic line, depth-dependent line state, and solver grid.

All arrays use CGS units. Index 0 is the top; z decreases with depth.
Ray directions are (sqrt(1-mu**2)*cos(phi), sqrt(1-mu**2)*sin(phi), mu),
with mu > 0 outward. B is an [N, 3] Cartesian vector in the same frame.
J^K_Q and rho_matrix are indexed [depth, K, Q+2], Q=-2..2.

Construct a LineState(atm, line, n_l, C_ul, D2) for each line in an
atmosphere, then Grid.build(atm, line, line_state).allocate(). Supply a
tensor using grid.set_JKQ(), or use grid_tensors.update_jkq_from_stokes()
with a previously calculated Stokes field. LineModel.calc_j(grid) evaluates
the line emissivity; LineModel.calc_S(j, chi) divides it by a prescribed
unpolarized opacity, and Grid.calc_tau(chi) calculates vertical optical depth
for that same opacity. Grid.solve_rte_fs(source, tau) performs a formal I/Q/U
solution for a fixed source; Grid.solve_rte_sc(chi) iterates the source and the
radiation tensor to self-consistency (Lambda iteration with short characteristics).
"""
import warnings
from dataclasses import dataclass, fields

import numpy as np

K_B = 1.380649e-16  # erg/K
AMU = 1.66053907e-24  # g
C_LIGHT = 2.99792458e10  # cm/s


@dataclass
class LambdaIterationResult:
    """Outcome of Grid.solve_rte_sc.

    history holds, per iteration, the largest change of any J^K_Q relative to
    the local J^0_0.
    """
    iterations: int
    converged: bool
    history: np.ndarray


def _frequency_opacity(chi, nnu):
    chi = np.asarray(chi)
    if chi.shape != (nnu,) or not np.isrealobj(chi):
        raise ValueError(f"chi must be real with shape ({nnu},)")
    if not np.all(np.isfinite(chi)) or np.any(chi <= 0):
        raise ValueError("chi must be finite and strictly positive")
    return chi


@dataclass
class Atmosphere1D:
    """Physical atmosphere, independent of the line; velocities are in cm/s.

    B is the mean field in gauss. B_unres is the optional strength in gauss
    of an isotropic unresolved component added to that mean. chi_c is pure
    thermal continuum opacity [cm^-1], constant across the line for now.
    rho [g/cm^3] and p [dyn/cm^2] may be absent.
    """
    z: np.ndarray
    T: np.ndarray
    v_z: np.ndarray
    B: np.ndarray
    rho: np.ndarray | None = None
    p: np.ndarray | None = None
    v_turb: np.ndarray | None = None
    v_x: np.ndarray | None = None
    v_y: np.ndarray | None = None
    B_unres: np.ndarray | None = None
    chi_c: np.ndarray | None = None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                setattr(self, field.name, np.asarray(value, dtype=float))
        if self.z.ndim != 1 or self.z.size == 0:
            raise ValueError("z must be a non-empty one-dimensional depth array")
        n = len(self.z)
        for name in ("v_turb", "v_x", "v_y", "B_unres", "chi_c"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(n))
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            expected = (n, 3) if field.name == "B" else (n,)
            if value.shape != expected or not np.all(np.isfinite(value)):
                raise ValueError(f"{field.name} must be finite with shape {expected}")
        if np.any(np.diff(self.z) >= 0):
            raise ValueError("z must strictly decrease with index (index 0 is the top)")
        if np.any(self.T <= 0):
            raise ValueError("T must be positive")
        for name in ("rho", "p", "v_turb", "B_unres", "chi_c"):
            value = getattr(self, name)
            if value is not None and np.any(value < 0):
                raise ValueError(f"{name} must be non-negative")

    @property
    def N(self):
        return len(self.z)

    def magnetic_angles(self):
        """Return field strength, inclination and azimuth; zero field uses zero angles."""
        bx, by, bz = self.B.T
        horizontal = np.hypot(bx, by)
        return np.hypot(horizontal, bz), np.arctan2(horizontal, bz), np.arctan2(by, bx)

    def projected_velocity(self, mu, phi):
        """Return v dot n [depth, mu, phi] for the outward-vertical frame."""
        mu = np.asarray(mu)
        phi = np.asarray(phi)
        horizontal = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
        return (self.v_z[:, None, None] * mu[None, :, None]
                + horizontal[None, :, None]
                * (self.v_x[:, None, None] * np.cos(phi)[None, None, :]
                   + self.v_y[:, None, None] * np.sin(phi)[None, None, :]))

    def point(self, k):
        """Return the physical fields at depth k, including the Cartesian field."""
        return {f.name: None if getattr(self, f.name) is None else getattr(self, f.name)[k]
                for f in fields(self) if f.name != "z"}


@dataclass
class LineModel:
    """Fixed atomic transition parameters; nu0 is in Hz and mass in atomic mass units."""
    nu0: float
    A_ul: float
    g_u: float
    g_l: float
    J_u: float
    J_l: float
    mass_amu: float

    def __post_init__(self):
        values = (self.nu0, self.A_ul, self.g_u, self.g_l, self.J_u, self.J_l, self.mass_amu)
        if not np.all(np.isfinite(values)) or self.nu0 <= 0 or self.A_ul <= 0 or self.mass_amu <= 0:
            raise ValueError("line parameters must be finite, with positive nu0, A_ul and mass")

    @property
    def lambda0(self):
        """Line-center wavelength [cm]."""
        return C_LIGHT / self.nu0

    def doppler_width(self, T, v_turb=0.0):
        """Thermal plus microturbulent Doppler width in frequency [Hz]."""
        v2 = 2 * K_B * np.asarray(T) / (self.mass_amu * AMU) + np.asarray(v_turb) ** 2
        return self.nu0 * np.sqrt(v2) / C_LIGHT

    def calc_j(self, grid, gamma=0.0, n_theta=8, n_chi=8, stimulated=True):
        """Return line emissivity j_I, j_Q, j_U on an allocated grid.

        The result has shape [depth, 3, frequency, mu, phi] and units
        erg s^-1 cm^-3 Hz^-1 sr^-1. It excludes continuum emission and does
        not modify the grid. JKQ must be supplied in the vertical frame.
        The existing emission calculation supports J_l=0 -> J_u=1 only.
        gamma is the Stokes reference angle; n_theta and n_chi set the
        unresolved-field angular quadrature.
        """
        if grid.line is not self:
            raise ValueError("grid must belong to this line")
        if self.J_l != 0 or self.J_u != 1:
            raise ValueError("calc_j supports only J_l=0 -> J_u=1")
        from grid_opacity import line_emissivity
        return line_emissivity(grid, gamma=gamma, n_theta=n_theta, n_chi=n_chi,
                               stimulated=stimulated)

    def calc_j_derivative(
        self, grid, gamma=0.0, n_theta=8, n_chi=8,
        stimulated=True, step=1e-3,
    ):
        """Return local d(j_I, j_Q, j_U)/d(B_x, B_y, B_z).

        Shape: [depth, field_component, stokes, frequency, mu, phi].
        Field components are in gauss; grid.JKQ is held fixed.
        """
        from dataclasses import replace

        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive")

        derivatives = []
        for component in range(3):
            samples = []
            for sign in (-1.0, 1.0):
                B = grid.atm.B.copy()
                B[:, component] += sign * step

                atm = replace(grid.atm, B=B)
                state = replace(grid.line_state, atm=atm)
                perturbed = replace(grid, atm=atm, line_state=state)
                perturbed.JKQ = grid.JKQ

                samples.append(self.calc_j(
                    perturbed, gamma=gamma, n_theta=n_theta, n_chi=n_chi,
                    stimulated=stimulated,
                ))

            derivatives.append((samples[1] - samples[0]) / (2 * step))

        return np.stack(derivatives, axis=1)

    @staticmethod
    def calc_S(j, chi):
        """Return S_I, S_Q, S_U = j / chi for a prescribed unpolarized opacity.

        j is the line emissivity from calc_j, shaped [depth, 3, frequency,
        mu, phi]. chi [cm^-1] must be finite, strictly positive, and have
        shape [frequency]; the same fixed opacity applies at all depths and
        directions and to all three components. The result has the shape of
        j and units erg s^-1 cm^-2 Hz^-1 sr^-1. No continuum emission is
        added and neither input is modified.
        """
        j = np.asarray(j)
        if j.ndim != 5 or j.shape[1] != 3 or not np.isrealobj(j) or not np.all(np.isfinite(j)):
            raise ValueError("j must be real and finite with shape [depth, 3, frequency, mu, phi]")
        chi = _frequency_opacity(chi, j.shape[2])
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return j / chi[None, None, :, None, None]


@dataclass
class LineState:
    """Depth-dependent properties of one line in one atmosphere.

    n_l [cm^-3], C_ul [s^-1], and D2 [s^-1] must be supplied. Optional a
    is the dimensionless Voigt damping at each depth. If absent, only natural
    broadening A_ul/(4*pi*delta_nu_D) is included. Derived properties are
    recomputed on access so changes to the atmosphere cannot leave stale values.
    """
    atm: Atmosphere1D
    line: LineModel
    n_l: np.ndarray
    C_ul: np.ndarray
    D2: np.ndarray
    a: np.ndarray | None = None

    def __post_init__(self):
        for name in ("n_l", "C_ul", "D2", "a"):
            value = getattr(self, name)
            if value is None and name == "a":
                continue
            value = np.asarray(value, dtype=float)
            if value.shape != (self.atm.N,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be finite, non-negative and have shape ({self.atm.N},)")
            setattr(self, name, value)

    @property
    def delta_nu_D(self):
        return self.line.doppler_width(self.atm.T, self.atm.v_turb)

    @property
    def damping(self):
        return self.a if self.a is not None else self.line.A_ul / (4 * np.pi * self.delta_nu_D)

    def destruction_probability(self, stimulated=True):
        """Depth-dependent photon destruction probability."""
        from grid_opacity import destruction_probability
        return destruction_probability(self.line, self.atm.T, self.C_ul, stimulated)

    @property
    def eps(self):
        """Photon destruction probability with stimulated emission enabled."""
        return self.destruction_probability()


@dataclass
class Grid:
    """A line state, angular/frequency quadrature, and mutable radiation state."""
    atm: Atmosphere1D
    line: LineModel
    line_state: LineState
    mu: np.ndarray
    w_mu: np.ndarray
    phi: np.ndarray
    w_phi: np.ndarray
    nu: np.ndarray  # offsets from line center [Hz]
    w_nu: np.ndarray

    def __post_init__(self):
        if self.line_state.atm is not self.atm or self.line_state.line is not self.line:
            raise ValueError("line_state must belong to this grid's atmosphere and line")

    @classmethod
    def build(cls, atm, line, line_state, n_mu_half=4, n_phi=8, n_nu=61, nu_max_doppler=6.0):
        """Gauss-Legendre mu, uniform phi and frequency, covering bulk velocity shifts."""
        x, w = np.polynomial.legendre.leggauss(n_mu_half)
        x, w = 0.5 * (x + 1.0), 0.5 * w
        mu, w_mu = np.concatenate([-x[::-1], x]), np.concatenate([w[::-1], w])
        phi = 2 * np.pi * np.arange(n_phi) / n_phi
        w_phi = np.full(n_phi, 2 * np.pi / n_phi)
        velocity = np.sqrt(atm.v_x**2 + atm.v_y**2 + atm.v_z**2)
        nu_extent = nu_max_doppler * np.max(line_state.delta_nu_D) + line.nu0 * np.max(velocity) / C_LIGHT
        nu = np.linspace(-nu_extent, nu_extent, n_nu)
        w_nu = np.full(n_nu, nu[1] - nu[0])
        w_nu[[0, -1]] *= 0.5
        return cls(atm, line, line_state, mu, w_mu, phi, w_phi, nu, w_nu)

    def allocate(self):
        """Allocate radiation and coefficient arrays; JKQ is initially zero."""
        n, nmu, nphi, nnu = self.atm.N, len(self.mu), len(self.phi), len(self.nu)
        self.tau = np.zeros((n, nnu))
        self.JKQ = np.zeros((n, 3, 5), dtype=complex)
        self.rho_matrix = np.zeros((n, 3, 5), dtype=complex)
        self.emis = np.zeros((n, 4, nnu, nmu, nphi))
        self.eta = np.zeros((n, 7, nnu, nmu, nphi))
        self.stokes = np.zeros((n, 4, nnu, nmu, nphi))
        return self

    def calc_tau(self, chi):
        """Store and return vertical optical depth [depth, frequency].

        chi [cm^-1] is the fixed, unpolarized opacity at each frequency,
        shared across all depths and directions, as in LineModel.calc_S.
        The integral runs from z to the top grid point, where tau is zero.
        z is in cm; tau is dimensionless and does not include a ray-angle
        factor or opacity above the top grid point.
        """
        chi = _frequency_opacity(chi, len(self.nu))
        z = self.atm.z
        if z.shape != (self.atm.N,) or not np.all(np.isfinite(z)) or np.any(np.diff(z) >= 0):
            raise ValueError("z must be finite and strictly decrease with index")
        with np.errstate(over="raise", invalid="raise"):
            tau = (z[0] - z[:, None]) * chi[None, :]
        self.tau = tau
        return self.tau

    def _default_boundaries(self):
        """Dark top and unpolarized Planck bottom (at nu0, bottom temperature) boundaries."""
        from grid_opacity import planck

        shape = (3, len(self.nu), len(self.mu), len(self.phi))
        top, bottom = np.zeros(shape), np.zeros(shape)
        bottom[0] = planck(self.line.nu0, self.atm.T[-1])
        return top, bottom

    def solve_rte_fs(self, source, tau, top=None, bottom=None):
        """Formal solution of I/Q/U transfer for a fixed source; stores it in grid.stokes.

        source [depth, 3, frequency, mu, phi] and tau [depth, frequency]
        can be obtained from line.calc_S and grid.calc_tau using the same chi.
        The source is interpolated linearly between depth nodes. Incoming top
        and bottom boundaries have shape [3, frequency, mu, phi]; by default
        the top is dark and the bottom is unpolarized Planck radiation at nu0
        and the bottom temperature. mu > 0 enters from the bottom. Returns
        [depth, 3, frequency, mu, phi]; grid.stokes V is left unchanged.
        The source is not updated from the result; see solve_rte_sc.
        """
        from formal_solver import solve_unpolarized_transfer

        default_top, default_bottom = self._default_boundaries()
        top = default_top if top is None else top
        bottom = default_bottom if bottom is None else bottom
        expected = (self.atm.N, 3, len(self.nu), len(self.mu), len(self.phi))
        if np.shape(source) != expected:
            raise ValueError(f"source must have shape {expected}")
        result = solve_unpolarized_transfer(source, tau, self.mu, top, bottom)
        self.stokes[:, :3] = result
        return result

    def solve_rte_sc(self, chi, top=None, bottom=None, JKQ0=None, gamma=0.0,
                     n_theta=8, n_chi=8, stimulated=True, tol=1e-6, max_iter=500):
        """Self-consistent I/Q/U solution by Lambda iteration with short characteristics.

        Each iteration computes the emissivity from grid.JKQ (line.calc_j),
        the source S = j / chi (line.calc_S), solves the transfer equations
        with parabolic short characteristics, and replaces grid.JKQ by the
        profile-weighted moments of the new Stokes field. chi [cm^-1, shape
        (frequency,)] is the fixed unpolarized opacity, also used for tau
        (calc_tau). S is the line source function only if chi equals the line
        opacity eta_I; this is not checked. Boundaries are as in solve_rte_fs.
        JKQ0 is the starting tensor [depth, 3, 5]; by default J^0_0 is the
        Planck function at nu0 and J^2_Q is zero. gamma is the Stokes
        reference angle. Convergence is the largest change of any J^K_Q
        relative to the local J^0_0 and must fall below tol.
        Plain Lambda iteration converges slowly in optically thick,
        weakly collisional (small eps) media, so max_iter may be reached.
        Stores I/Q/U in grid.stokes (V is untouched) and the tensor in
        grid.JKQ, and returns a LambdaIterationResult.
        """
        from formal_solver import solve_short_characteristics
        from grid_opacity import planck
        from grid_tensors import absorption_profile, jkq_from_stokes, tensor_basis

        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        shape = (self.atm.N, 3, len(self.nu), len(self.mu), len(self.phi))
        default_top, default_bottom = self._default_boundaries()
        top = default_top if top is None else top
        bottom = default_bottom if bottom is None else bottom
        tau = self.calc_tau(chi)
        if JKQ0 is None:
            JKQ0 = np.zeros((self.atm.N, 3, 5), dtype=complex)
            JKQ0[:, 0, 2] = planck(self.line.nu0, self.atm.T)
        self.set_JKQ(JKQ0)

        basis = tensor_basis(self.mu, self.phi, gamma=gamma)
        profile = absorption_profile(self)
        stokes = np.zeros((self.atm.N, 4) + shape[2:])
        history = []
        converged = False
        for _ in range(int(max_iter)):
            j = self.line.calc_j(self, gamma=gamma, n_theta=n_theta, n_chi=n_chi,
                                 stimulated=stimulated)
            stokes[:, :3] = solve_short_characteristics(
                self.line.calc_S(j, chi), tau, self.mu, top, bottom)
            updated = jkq_from_stokes(self, stokes, basis=basis, profile=profile)
            scale = np.maximum(np.abs(updated[:, 0, 2]), np.finfo(float).tiny)
            history.append(np.max(np.abs(updated - self.JKQ) / scale[:, None, None]))
            self.set_JKQ(updated)
            if history[-1] < tol:
                converged = True
                break
        self.stokes[:, :3] = stokes[:, :3]
        if not converged:
            warnings.warn(
                f"Lambda iteration did not converge in {len(history)} iterations "
                f"(last relative change {history[-1]:.3e} >= tol {tol:.1e})",
                RuntimeWarning, stacklevel=2)
        return LambdaIterationResult(len(history), converged, np.array(history))

    def set_JKQ(self, JKQ):
        """Set a prescribed depth-dependent radiation tensor [N, 3, 5]."""
        value = np.asarray(JKQ, dtype=complex)
        if value.shape != (self.atm.N, 3, 5) or not np.all(np.isfinite(value)):
            raise ValueError(f"JKQ must be finite with shape ({self.atm.N}, 3, 5)")
        self.JKQ[...] = value
        return self.JKQ
