"""Compute and plot emergent spectra starting from a wavelength grid.

Run: python plot_physical_flow.py [model.txt] [--solver delo]
Use --help for wavelength-grid, velocity, and output options.

The grid is linear in vacuum wavelength; --lambda-range-nm MIN MAX sets its
endpoints. Otherwise it spans +/- --extent minimum local Doppler widths about
line center. --spectral-axis selects absolute frequency, wavelength, or both
(default) for spectral plots. Wavelength plots show I_lambda per nm; frequency
plots show I_nu per Hz. Q/I and U/I are percentages in all plots.
Plotted V/I values with magnitude below 1e-15 percent are shown as zero;
the calculated Stokes vector is unchanged.
Plot filenames and titles also show the resolved field strength, inclination
from the outward vertical, and azimuth from +x towards +y, in degrees. The
azimuth is undefined for a vertical field; both angles are undefined at B=0.

This is a fixed-source calculation using Jbar00, Jbar20 and optional Jbar21
and Jbar22 components from the model, not a self-consistent NLTE calculation.
The optional columns Jbar21_re, Jbar21_im, Jbar22_re, Jbar22_im are in the
same units and vertical frame as Jbar00 and Jbar20; omitted columns are zero.
The default scalar solver treats Stokes
components independently; --solver delo uses the polarized DELO-linear solver
and propagates all four Stokes components with the full propagation matrix.
The file supplies populations. A fixed-population two-level approximation uses
the prescribed rank-2 moments to pump upper-level alignment, with Hanle relaxation by A_ul, C_ul and
the supplied D2. Jbar00 does not directly set emission in this population-based
branch: n_upper sets isotropic emission. It neglects stimulated-emission effects on alignment and
polarized propagation, and alignment pumping or relaxation by other transitions.
The supplied example moments are from an unmagnetized model; prescribed moments
remain fixed with B.
The rank-2 population emissivity no longer uses a thermal stimulated factor,
including at B = 0. Zeeman changes to the profile-weighted pumping are neglected.
--b-field BX BY BZ sets a constant Cartesian field in gauss, with z outward;
--b-unres sets the strength of an isotropic, microstructured component added to
that field. --d2 sets the elastic depolarization rate in s^-1 (default zero);
Q_elast is a broadening rate and is not used as D2. Nonzero fields require DELO.
The bulk velocity is a constant vertical velocity using a nonrelativistic Doppler shift. By
default the total line-center optical depth is taken from log_tau_line. A gray,
thermal continuum is inferred by subtracting the population-based line opacity
from the reconstructed line-center opacity. This assumes all residual opacity
is continuum and treats scattering as thermal emission. The inversion is
sensitive to rounding of the tabulated optical depths. --line-only omits this
continuum. The continuum thermal emission and lower boundary use the Planck
function at each wavelength; the line source function uses nu0. A nonzero
velocity does not update the prescribed JKQ or inferred static continuum.
"""
import argparse
import os

import numpy as np

from atmosphere import AMU, C_LIGHT, K_B, Atmosphere1D, Grid, LineModel, LineState
from formal_solver import solve_delo, solve_short_characteristics
from grid_opacity import fill_grid, planck
from plot_model_file import (
    DEFAULT_MODEL, M3_TO_CM3, W_TO_CGS, build_grid, continuum_from_tau500, cumulative_tau,
    format_spectral_axis, header_value, plt, radiation_tensors_from_model, read_model,
    resolve_model, stokes_percent,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_physical_flow")


def model_parameters(col, header, mass):
    """Derive line and microturbulence from the model; v_turb has shape [depth]."""
    lam0 = header_value(header, r"lambda0\s*=\s*([0-9.eE+-]+)\s*nm", None, "lambda0") * 1e-7
    a_ul = header_value(header, r"A_ul\s*=\s*([0-9.eE+-]+)", None, "A_ul")
    gamma_rad = header_value(header, r"Gamma_rad\s*=\s*([0-9.eE+-]+)", None, "Gamma_rad")
    line = LineModel(C_LIGHT / lam0, a_ul, 1.0, 1.0, 1, 0, mass)
    q_elast = col.get("Q_elast", np.zeros_like(col["T_K"]))
    v_d2 = ((gamma_rad + q_elast) * lam0 / (4 * np.pi * col["a_damp"])) ** 2
    v_turb2 = v_d2 - 2 * K_B * col["T_K"] / (mass * AMU)
    if np.any(v_turb2 < -1e-3 * v_d2):
        raise ValueError("a_damp implies a Doppler width below the thermal one; check --mass")
    return line, np.sqrt(np.maximum(v_turb2, 0.0))


def wavelength_grid(line, delta_nu_D, n_lambda, extent, limits_nm):
    """Return increasing wavelengths [cm] and the reference width [cm]."""
    if n_lambda < 3 or n_lambda % 2 == 0:
        raise ValueError("--n-lambda must be odd and at least 3")
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("--extent must be finite and positive")
    delta_lambda_ref = line.lambda0 * np.min(delta_nu_D) / line.nu0
    if limits_nm is None:
        limits = np.array([line.lambda0 - extent * delta_lambda_ref,
                           line.lambda0 + extent * delta_lambda_ref])
    else:
        limits = np.asarray(limits_nm) * 1e-7
    if not np.all(np.isfinite(limits)) or limits[0] <= 0 or limits[0] >= limits[1]:
        raise ValueError("wavelength limits must be finite, positive, and increasing")
    return np.linspace(*limits, n_lambda), delta_lambda_ref


def continuum_from_line_center(col, line, v_turb):
    """Reconstruct nodal gray continuum from total line-center optical depth."""
    if "log_tau_line" not in col or "n_upper" not in col:
        raise ValueError("model must contain log_tau_line and n_upper")
    total_tau = 10.0 ** col["log_tau_line"]
    dz = -np.diff(col["height_km"] * 1e5)
    if (not np.all(np.isfinite(total_tau)) or np.any(total_tau <= 0)
            or np.any(np.diff(total_tau) <= 0) or np.any(dz <= 0)):
        raise ValueError("line-center optical depth must be finite, positive, and strictly increasing")
    grid = build_grid(col, line, v_turb, np.array([1.]), np.array([0.]),
                      np.array([1.]), np.zeros(len(total_tau)), 0., use_populations=True)
    total_chi = np.empty_like(total_tau)
    total_chi[0] = 2 * total_tau[0] / dz[0]
    for k in range(len(dz)):
        total_chi[k + 1] = 2 * (total_tau[k + 1] - total_tau[k]) / dz[k] - total_chi[k]
    chi_c = total_chi - grid.eta[:, 0, 0, 0, 0]
    if not np.all(np.isfinite(chi_c)) or np.any(chi_c < 0):
        raise ValueError("line-center optical depths imply negative or nonfinite continuum opacity; "
                         "the tabulated depths may be too rounded or inconsistent with the line model")
    return chi_c


def physical_flow(col, line, v_turb, wavelength, mu, v_z=0.0, continuum=True, gamma=0.0,
                  solver="sc", B=None, B_unres: float | np.ndarray = 0.0,
                  D2: float | np.ndarray = 0.0, n_theta=8, n_chi=8):
    """Return frequency offsets, reduced x, source, optical depth, and Stokes.

    Wavelength [NL] is in cm; mu [Nmu] contains outward ray cosines.
    x and tau are [ND, NL, Nmu, 1]; source is [ND, 3, NL, Nmu, 1].
    Stokes is [ND, 3, NL, Nmu, 1] for solver="sc" and
    [ND, 4, NL, Nmu, 1] for solver="delo". Radiation quantities are per Hz in CGS.
    v_z is in cm/s; positive is upward. The last dimension is azimuth.
    B is a Cartesian field in gauss, either [3] or [ND, 3]. B_unres [G] and
    D2 [s^-1] are scalars or [ND]; the unresolved direction quadrature is
    n_theta by n_chi. Nonzero fields require solver="delo".
    With polarized propagation, source is j[:3]/eta_I, not K^-1 j.
    The tabulated half-cell is included in tau, but the boundary of the formal
    solution is at the top grid point; no transfer through that half-cell is solved.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if (wavelength.ndim != 1 or len(wavelength) < 3
            or not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0)
            or np.any(np.diff(wavelength) <= 0)):
        raise ValueError("wavelength must be a positive, strictly increasing grid of at least 3 points")
    if mu.ndim != 1 or not len(mu) or not np.all(np.isfinite(mu)) or np.any((mu <= 0) | (mu > 1)):
        raise ValueError("mu must contain finite outward cosines in (0, 1]")
    if not np.isfinite(v_z) or abs(v_z) >= C_LIGHT:
        raise ValueError("v_z must be finite and have magnitude below the speed of light")
    if solver not in ("sc", "delo"):
        raise ValueError("solver must be 'sc' or 'delo'")

    n = len(col["height_km"])
    if "n_upper" not in col:
        raise ValueError("model must contain n_upper")
    B = np.zeros((n, 3)) if B is None else np.asarray(B, dtype=float)
    if B.shape == (3,):
        B = np.broadcast_to(B, (n, 3))
    B_unres = np.broadcast_to(np.asarray(B_unres, dtype=float), (n,))
    D2 = np.broadcast_to(np.asarray(D2, dtype=float), (n,))
    if not isinstance(n_theta, int) or not isinstance(n_chi, int) or n_theta < 1 or n_chi < 1:
        raise ValueError("n_theta and n_chi must be positive integers")
    if solver != "delo" and (np.any(B != 0) or np.any(B_unres != 0)):
        raise ValueError("nonzero magnetic fields require solver='delo'")
    chi_c = continuum_from_line_center(col, line, v_turb) if continuum else np.zeros(n)
    atm = Atmosphere1D(z=col["height_km"] * 1e5, T=col["T_K"],
                       v_z=np.full(n, v_z), B=B, B_unres=B_unres,
                       v_turb=v_turb, chi_c=chi_c)
    state = LineState(atm, line, col["n_lower"] * M3_TO_CM3, col["C_ul"],
                      D2, a=col["a_damp"], n_u=col["n_upper"] * M3_TO_CM3)

    # Wavelength increases, so the corresponding frequency-offset grid decreases.
    # Frequency weights are positive trapezoidal widths; prescribed JKQ means
    # they are not used in this fixed-source calculation.
    nu = C_LIGHT / wavelength - line.nu0  # [NL], Hz
    dnu = -np.diff(nu)
    w_nu = np.empty_like(nu)
    w_nu[0], w_nu[-1] = dnu[0] / 2, dnu[-1] / 2
    w_nu[1:-1] = (dnu[:-1] + dnu[1:]) / 2
    grid = Grid(atm, line, state, mu, np.full(len(mu), 2 / len(mu)),
                np.zeros(1), np.array([2 * np.pi]), nu, w_nu).allocate()
    grid.set_JKQ(radiation_tensors_from_model(col))

    # Positive projected velocity is toward the observer; recession is -v_proj.
    v_proj = atm.projected_velocity(mu, grid.phi)  # [ND, Nmu, 1]
    delta_lambda = line.lambda0 * state.delta_nu_D / line.nu0
    x = (wavelength[None, :, None, None]
         - line.lambda0 * (1 - v_proj[:, None] / C_LIGHT))
    x = x / delta_lambda[:, None, None, None]  # [ND, NL, Nmu, 1]
    fill_grid(grid, gamma=gamma, n_theta=n_theta, n_chi=n_chi, wavelength_profile=True)

    # eta_I is [ND, NL, Nmu, 1], j and S are [ND, 3, NL, Nmu, 1].
    eta_i = grid.eta[:, 0]
    if not np.all(np.isfinite(eta_i)) or np.any(eta_i <= 0):
        raise ValueError("total opacity must be finite and positive at every grid point")
    source = grid.emis[:, :3] / eta_i[:, None]
    # Integrate vertical opacity separately for each ray; the solver divides
    # each optical-depth increment by |mu|.
    tau = np.empty_like(eta_i)
    stokes = np.empty((n, 4 if solver == "delo" else 3, len(nu), len(mu), 1))
    dz = -np.diff(atm.z)
    if solver == "delo":
        top = np.zeros((4, len(nu), len(mu), 1))
        bottom = np.zeros_like(top)
        bottom[0] = planck(C_LIGHT / wavelength, atm.T[-1])[:, None, None]
        stokes = solve_delo(grid.emis, grid.eta, atm.z, mu, top, bottom)
        for k in range(len(mu)):
            tau[:, :, k, 0] = (
                cumulative_tau(eta_i[:, :, k, 0], dz) + eta_i[0, :, k, 0] * dz[0] / 2
            )
    else:
        for k, ray in enumerate(mu):
            opacity = eta_i[:, :, k, 0]
            tau[:, :, k, 0] = cumulative_tau(opacity, dz) + opacity[0] * dz[0] / 2
            boundary = np.zeros((3, len(nu), 1, 1))
            boundary[0] = planck(C_LIGHT / wavelength, atm.T[-1])[:, None, None]
            stokes[:, :, :, k:k + 1, :] = solve_short_characteristics(
                source[:, :, :, k:k + 1, :], tau[:, :, k, 0], np.array([ray]),
                np.zeros_like(boundary), boundary)
    return nu, x, source, tau, stokes


def plot_spectra(wavelength, nu, stokes, mu, line, delta_lambda_ref, outdir, continuum,
                 solver="sc", field_label="", B=None, spectral_axis="both"):
    """Plot top-boundary I and Q/I, U/I, V/I against physical spectral coordinates."""
    if spectral_axis not in ("frequency", "wavelength", "both"):
        raise ValueError("spectral_axis must be 'frequency', 'wavelength', or 'both'")
    os.makedirs(outdir, exist_ok=True)
    wavelength_nm = wavelength * 1e7
    prefix = f"{solver}_{'continuum' if continuum else 'line'}"
    field_suffix = ""
    orientation_label = ""
    if B is not None:
        B = np.asarray(B, dtype=float)
        if B.shape != (3,) or not np.all(np.isfinite(B)):
            raise ValueError("B must be a finite Cartesian vector with shape (3,)")
        horizontal = np.hypot(B[0], B[1])
        strength = np.hypot(horizontal, B[2])
        inclination = np.degrees(np.arctan2(horizontal, B[2])) if strength else None
        azimuth = np.degrees(np.arctan2(B[1], B[0])) if horizontal else None
        inc = f"{inclination:.6f}" if inclination is not None else "NA"
        azi = f"{azimuth:.6f}" if azimuth is not None else "NA"
        field_suffix = f"_B{strength:.6f}G_inc{inc}deg_azi{azi}deg"
        orientation_label = (f"|B|={strength:.3f} G, inclination={inc} deg, "
                             f"azimuth={azi} deg")
        if inclination is None or azimuth is None:
            orientation_label += " (NA: undefined)"
    axes = (
        ("wavelength", wavelength_nm, "Wavelength [nm]", True),
        ("frequency", (line.nu0 + nu) / 1e9, "Frequency [GHz]", False),
    )
    for name, coordinate, xlabel, per_wavelength in axes:
        if spectral_axis != "both" and name != spectral_axis:
            continue
        fig, panels = plt.subplots(1, 4, figsize=(19, 5.2), constrained_layout=True)
        order = np.argsort(coordinate)
        for k, ray in enumerate(mu):
            # Convert CGS I_nu to SI; for I_lambda also multiply by |dnu/dlambda|
            # and convert the wavelength density from per cm to per nm.
            emergent = stokes[0, :, :, k, 0] / W_TO_CGS  # [3 or 4, NL], SI per Hz
            if per_wavelength:
                emergent = emergent * (C_LIGHT / wavelength**2 * 1e-7)[None, :]
            if emergent.shape[0] == 3:
                emergent = np.vstack((emergent, np.zeros((1, emergent.shape[1]))))
            values = stokes_percent(emergent)
            values[3, np.abs(values[3]) < 1e-15] = 0.0
            for i, ax in enumerate(panels):
                ax.plot(coordinate[order], values[i, order], label=f"mu={ray:g}")
                ax.set_xlabel(xlabel)
                format_spectral_axis(ax)
                ax.grid(alpha=0.3)
        panels[0].set_ylabel(
            r"$I_\lambda$ [W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]" if per_wavelength
            else r"$I_\nu$ [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
        panels[1].set_ylabel("Q/I [%]")
        panels[2].set_ylabel("U/I [%]")
        panels[3].set_ylabel("V/I [%]")
        panels[0].legend()
        fig.suptitle(
            "Emergent spectra (" + solver + "): "
            + ("line + continuum" if continuum else "line only")
            + (f"\n{field_label}" if field_label else "")
            + (f"\n{orientation_label}" if orientation_label else "")
        )
        path = os.path.join(outdir, f"{prefix}_{name}{field_suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--outdir", default=OUT)
    parser.add_argument("--n-lambda", type=int, default=241)
    parser.add_argument("--extent", type=float, default=12.0,
                        help="grid half-width in minimum local Doppler widths")
    parser.add_argument("--lambda-range-nm", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="explicit wavelength range in nm, instead of --extent")
    parser.add_argument("--mu", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.33, 0.5, 0.75, 1.0])
    parser.add_argument("--v-z", type=float, default=0.0,
                        help="constant vertical bulk velocity in km/s, positive upward")
    parser.add_argument("--solver", choices=("sc", "delo"), default="sc",
                        help="sc: uncoupled scalar formal solution; delo: polarized DELO-linear")
    parser.add_argument("--spectral-axis", choices=("frequency", "wavelength", "both"), default="both",
                        help="spectral plot x-axis: absolute frequency, wavelength, or both (default)")
    parser.add_argument("--b-field", type=float, nargs=3, metavar=("BX", "BY", "BZ"),
                        default=(0., 0., 0.), help="constant Cartesian magnetic field in gauss")
    parser.add_argument("--b-unres", type=float, default=0.,
                        help="strength in gauss of isotropic unresolved component")
    parser.add_argument("--d2", type=float, default=0.,
                        help="elastic depolarization rate in s^-1, not Q_elast")
    parser.add_argument("--n-theta", type=int, default=8, help="unresolved-field polar quadrature")
    parser.add_argument("--n-chi", type=int, default=8, help="unresolved-field azimuth quadrature")
    parser.add_argument("--mass", type=float, default=87.62, help="atomic mass in amu")
    parser.add_argument("--gamma-deg", type=float, default=90.0)
    opacity_group = parser.add_mutually_exclusive_group()
    opacity_group.add_argument("--continuum", dest="continuum", action="store_true",
                               help="include continuum inferred from log_tau_line (default)")
    opacity_group.add_argument("--line-only", dest="continuum", action="store_false",
                               help="omit the inferred continuum")
    parser.set_defaults(continuum=True)
    args = parser.parse_args()

    col, header = read_model(resolve_model(args.model))
    line, v_turb = model_parameters(col, header, args.mass)
    wavelength, width = wavelength_grid(
        line, line.doppler_width(col["T_K"], v_turb), args.n_lambda, args.extent, args.lambda_range_nm)
    mu = np.array(args.mu)
    nu, x, source, tau, stokes = physical_flow(
        col, line, v_turb, wavelength, mu, args.v_z * 1e5, args.continuum,
        np.deg2rad(args.gamma_deg), args.solver, B=args.b_field, B_unres=args.b_unres,
        D2=args.d2, n_theta=args.n_theta, n_chi=args.n_chi)
    print(f"x {x.shape}; source {source.shape}; tau {tau.shape}; Stokes {stokes.shape}")
    print(f"Fixed prescribed Jbar tensors; {args.solver} transfer; "
          f"B={tuple(args.b_field)} G, B_unres={args.b_unres:g} G, D2={args.d2:g} s^-1.")
    print("Fixed populations; two-level alignment neglects stimulated and multilevel alignment effects.")
    if args.continuum:
        print("Gray thermal continuum inferred from the static total line-center optical depth.")
        print("The half-cell optical depth is recorded but not traversed by the formal solution.")
        ratio = continuum_from_line_center(col, line, v_turb) / continuum_from_tau500(col)[1]
        print(f"chi_c(line-centre inversion) / chi_c(tau500): min {ratio.min():.3g}, "
              f"median {np.median(ratio):.3g}, max {ratio.max():.3g}")
    if args.v_z != 0:
        print("Bulk velocity does not update the prescribed radiation tensors.")
    field_label = (f"B={tuple(args.b_field)} G, B_unres={args.b_unres:g} G, "
                   f"D2={args.d2:g} s^-1")
    plot_spectra(wavelength, nu, stokes, mu, line, width, args.outdir, args.continuum,
                 args.solver, field_label=field_label, B=args.b_field, spectral_axis=args.spectral_axis)


if __name__ == "__main__":
    main()
