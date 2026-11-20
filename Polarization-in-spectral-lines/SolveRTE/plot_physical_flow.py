"""Compute and plot emergent spectra starting from a wavelength grid.

Run: python plot_physical_flow.py [model.txt] [--continuum]
Use --help for wavelength-grid, velocity, and output options.

The grid is linear in vacuum wavelength; --lambda-range-nm MIN MAX sets its
endpoints. Otherwise it spans +/- --extent minimum local Doppler widths about
line center. The output contains wavelength, frequency-offset, and reduced-
wavelength plots. Wavelength plots show I_lambda per nm; frequency plots
show I_nu per Hz. Q/I and U/I are percentages in all plots.
Plot titles show the minimum local Doppler width in wavelength and frequency.

This is a fixed-source calculation using Jbar00 and Jbar20 from the model,
not a self-consistent NLTE calculation. The line is unmagnetized, the bulk
velocity is a constant vertical velocity using a nonrelativistic Doppler shift,
and the formal solution neglects polarization-dependent propagation. With
--continuum, the continuum is gray and thermal, inferred from tau500 as in
plot_model_file.py. A nonzero velocity does not update the prescribed JKQ.
"""
import argparse
import os

import numpy as np

from atmosphere import AMU, C_LIGHT, K_B, Atmosphere1D, Grid, LineModel, LineState
from formal_solver import solve_short_characteristics
from grid_opacity import fill_grid, planck
from plot_model_file import (
    DEFAULT_MODEL, M3_TO_CM3, W_TO_CGS, continuum_from_tau500, cumulative_tau,
    header_value, plt, read_model, resolve_model, stokes_percent,
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


def physical_flow(col, line, v_turb, wavelength, mu, v_z=0.0, continuum=False, gamma=0.0):
    """Return frequency offsets, reduced x, source, optical depth, and Stokes.

    Wavelength [NL] is in cm; mu [Nmu] contains outward ray cosines.
    x and tau are [ND, NL, Nmu, 1]; source and Stokes are
    [ND, 3, NL, Nmu, 1]. Radiation quantities are per Hz in CGS.
    v_z is in cm/s; positive is upward. The last dimension is azimuth.
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

    n = len(col["height_km"])
    tau_cont, chi_c = continuum_from_tau500(col) if continuum else (np.zeros(n), np.zeros(n))
    atm = Atmosphere1D(z=col["height_km"] * 1e5, T=col["T_K"],
                       v_z=np.full(n, v_z), B=np.zeros((n, 3)),
                       v_turb=v_turb, chi_c=chi_c)
    state = LineState(atm, line, col["n_lower"] * M3_TO_CM3, col["C_ul"],
                      np.zeros(n), a=col["a_damp"])

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
    jkq = np.zeros((n, 3, 5), complex)  # [ND, K, Q+2]
    jkq[:, 0, 2] = col["Jbar00"] * W_TO_CGS
    jkq[:, 2, 2] = col["Jbar20"] * W_TO_CGS
    grid.set_JKQ(jkq)

    # x is computed in the comoving frame for each depth, wavelength and ray.
    # fill_grid evaluates the same x internally to build the LL04 Phi tables,
    # then combines Phi with the prescribed JKQ to obtain j and eta.
    v_proj = atm.projected_velocity(mu, grid.phi)  # [ND, Nmu, 1]
    x = (-nu[None, :, None, None] + line.nu0 * v_proj[:, None] / C_LIGHT)
    x = x / state.delta_nu_D[:, None, None, None]  # [ND, NL, Nmu, 1]
    fill_grid(grid, gamma=gamma)

    # eta_I is [ND, NL, Nmu, 1], j and S are [ND, 3, NL, Nmu, 1].
    eta_i = grid.eta[:, 0]
    if not np.all(np.isfinite(eta_i)) or np.any(eta_i <= 0):
        raise ValueError("total opacity must be finite and positive at every grid point")
    source = grid.emis[:, :3] / eta_i[:, None]
    line_opacity = eta_i - chi_c[:, None, None, None]
    if not np.all(np.isfinite(line_opacity)) or np.any(line_opacity < 0):
        raise ValueError("line opacity must be finite and non-negative")

    # Integrate vertical opacity from the top separately for each ray:
    # velocity shifts the line differently at different mu. tau500 supplies
    # the continuum integral, consistent with plot_model_file.py.
    tau = np.empty_like(eta_i)
    stokes = np.empty_like(source)
    dz = -np.diff(atm.z)
    for k, ray in enumerate(mu):
        tau[:, :, k, 0] = cumulative_tau(line_opacity[:, :, k, 0], dz) + tau_cont[:, None]
        boundary = np.zeros((3, len(nu), 1, 1))
        boundary[0] = planck(line.nu0, atm.T[-1])
        stokes[:, :, :, k:k + 1, :] = solve_short_characteristics(
            source[:, :, :, k:k + 1, :], tau[:, :, k, 0], np.array([ray]),
            np.zeros_like(boundary), boundary)
    return nu, x, source, tau, stokes


def plot_spectra(wavelength, nu, stokes, mu, line, delta_lambda_ref, outdir, continuum):
    """Plot top-boundary I and Q/I, U/I against three spectral coordinates."""
    os.makedirs(outdir, exist_ok=True)
    wavelength_nm = wavelength * 1e7
    delta_nu_ref = line.nu0 * delta_lambda_ref / line.lambda0
    prefix = "continuum" if continuum else "line"
    axes = (
        ("wavelength", wavelength_nm, "Wavelength [nm]", True),
        ("frequency", nu / 1e9, "Frequency offset from line center [GHz]", False),
        ("reduced_wavelength", (wavelength - line.lambda0) / delta_lambda_ref,
         "Wavelength offset [reference Doppler widths]", True),
    )
    for name, coordinate, xlabel, per_wavelength in axes:
        fig, panels = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        order = np.argsort(coordinate)
        for k, ray in enumerate(mu):
            # Convert CGS I_nu to SI; for I_lambda also multiply by |dnu/dlambda|
            # and convert the wavelength density from per cm to per nm.
            emergent = stokes[0, :, :, k, 0] / W_TO_CGS  # [3, NL], SI per Hz
            if per_wavelength:
                emergent = emergent * (C_LIGHT / wavelength**2 * 1e-7)[None, :]
            values = stokes_percent(emergent)
            for i, ax in enumerate(panels):
                ax.plot(coordinate[order], values[i, order], label=f"mu={ray:g}")
                ax.set_xlabel(xlabel)
                ax.grid(alpha=0.3)
        panels[0].set_ylabel(
            r"$I_\lambda$ [W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]" if per_wavelength
            else r"$I_\nu$ [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
        panels[1].set_ylabel("Q/I [%]")
        panels[2].set_ylabel("U/I [%]")
        panels[0].legend()
        fig.suptitle(
            "Emergent spectra: " + ("line + continuum" if continuum else "line only")
            + "\n"
            + rf"$\Delta\lambda_{{D,\mathrm{{ref}}}} = {delta_lambda_ref * 1e7:.5g}$ nm, "
            + rf"$\Delta\nu_{{D,\mathrm{{ref}}}} = {delta_nu_ref / 1e9:.5g}$ GHz"
        )
        path = os.path.join(outdir, f"{prefix}_{name}.png")
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
    parser.add_argument("--mu", type=float, nargs="+", default=[0.1, 0.33, 0.5, 0.75, 1.0])
    parser.add_argument("--v-z", type=float, default=0.0,
                        help="constant vertical bulk velocity in km/s, positive upward")
    parser.add_argument("--mass", type=float, default=87.62, help="atomic mass in amu")
    parser.add_argument("--gamma-deg", type=float, default=90.0)
    parser.add_argument("--continuum", action="store_true")
    args = parser.parse_args()

    col, header = read_model(resolve_model(args.model))
    line, v_turb = model_parameters(col, header, args.mass)
    wavelength, width = wavelength_grid(
        line, line.doppler_width(col["T_K"], v_turb), args.n_lambda, args.extent, args.lambda_range_nm)
    mu = np.array(args.mu)
    nu, x, source, tau, stokes = physical_flow(
        col, line, v_turb, wavelength, mu, args.v_z * 1e5, args.continuum, np.deg2rad(args.gamma_deg))
    print(f"x {x.shape}; source {source.shape}; tau {tau.shape}; Stokes {stokes.shape}")
    print("Fixed prescribed Jbar tensors; unmagnetized scalar-opacity formal solution.")
    if args.v_z != 0:
        print("Bulk velocity does not update the prescribed radiation tensors.")
    plot_spectra(wavelength, nu, stokes, mu, line, width, args.outdir, args.continuum)


if __name__ == "__main__":
    main()
