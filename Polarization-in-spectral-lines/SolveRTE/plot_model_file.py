"""Plot source functions and Stokes solutions for a tabulated model atmosphere.

Run with the `SLAB` conda environment (it needs numpy, scipy, numba, matplotlib):
    python plot_model_file.py [model.txt]
Without an argument it reads sr4607_falc_Jbar_anisotropy.txt next to this script.
Plots go to plots_model/ next to this script.

The text file gives, from the top of the atmosphere downwards, height_km,
log_tau500, T_K, log_tau_line, Jbar00, Jbar20 (W m^-2 Hz^-1 sr^-1),
n_lower and n_upper (m^-3),
C_ul, epsilon and a_damp; the line wavelength, A_ul and Gamma_rad are read from
its header. The model is unmagnetized and static. Elastic collisions (Q_elast)
only broaden the line (through a_damp) and are not treated as depolarizing.
Optional columns Jbar21_re, Jbar21_im, Jbar22_re, Jbar22_im provide the real
and imaginary parts of the vertical-frame rank-2 moments in the same units as
Jbar00 and Jbar20. Each absent column is zero; negative-Q components follow
J^2_{-Q} = (-1)^Q conj(J^2_Q).

Pipeline: the file's populations set the line opacity and isotropic emission;
two-level radiative pumping supplies the aligned contribution (neglecting
stimulated-emission effects on alignment and polarized propagation).
The prescribed moments are not iterated to self-consistency. Comparison plots
also show the original two-level epsilon/J source with the same opacity and
boundaries, isolating the effect of the source prescription.
Optical depths use the trapezoid rule, referenced to zero at the top grid point.
The file's log_tau_line is total line-centre optical depth including continuum,
with a half-cell top contribution, and is used only as a diagnostic.

Two cases are plotted (the second one has "continuum" in its file names):
  line only:         chi = eta_line;  S_I, S_Q and S_U do not depend on frequency.
  line + continuum:  chi = eta_line + chi_c, with an unpolarized thermal chi_c
                     inferred from the file: the optical depth is its tau500
                     (log_tau500 column) and chi_c = tau500 |d ln tau500 / dz|.
                     This assumes a gray, purely thermal continuum equal to the
                     total continuum opacity at 500 nm, including scattering.
Radiation quantities are plotted in SI (default, as in the file) or cgs units.
Stokes plots show I and 100 Q/I, 100 U/I; source-function plots remain absolute.
Anisotropy plots use the prescribed file moments, w = sqrt(2) Jbar20/Jbar00.
--spectral-axis selects absolute frequency, wavelength, or both for spectral
plots; the calculation still uses its frequency grid. Height and optical-depth
plots retain their depth axes. Wavelength-axis plots retain per-Hz intensities.
"""
import argparse
import os
import re
import sys
import warnings

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atmosphere import AMU, C_LIGHT, K_B, Atmosphere1D, Grid, LineModel, LineState
from grid_opacity import fill_grid, planck

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_model")
REQUIRED = ("height_km", "log_tau500", "T_K", "Jbar00", "Jbar20", "n_lower",
            "C_ul", "epsilon", "a_damp")
J_COLUMNS = ("Jbar00", "Jbar20", "Jbar21_re", "Jbar21_im", "Jbar22_re", "Jbar22_im")
W_TO_CGS = 1e3  # W m^-2 Hz^-1 sr^-1 -> erg s^-1 cm^-2 Hz^-1 sr^-1
M3_TO_CM3 = 1e-6
# factor from cgs to the plotted unit, and the unit label
UNITS = {
    "si": (1.0 / W_TO_CGS, r"W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$"),
    "cgs": (1.0, r"erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$"),
}


def read_model(path):
    """Return (columns, header text); columns are ordered from the top down."""
    names, rows, header = None, [], []
    with open(path) as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                header.append(text)
                if "height_km" in text:
                    names = text.lstrip("#").split()
                continue
            rows.append([float(item) for item in text.split()])
    if names is None or not rows:
        raise ValueError("no '# height_km ...' column-name line or no data rows found")
    if len(set(names)) != len(names):
        raise ValueError("model column names must be unique")
    if any(len(row) != len(names) for row in rows):
        raise ValueError("a data row does not match the number of column names")
    table = np.array(rows)
    model = {name: table[:, i] for i, name in enumerate(names)}
    missing = [name for name in REQUIRED if name not in model]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if len(table) < 2 or not np.all(np.isfinite(table)):
        raise ValueError("need at least two finite data rows")
    if np.any(np.diff(model["height_km"]) >= 0):
        raise ValueError("rows must go from the top down (height strictly decreasing)")
    return model, "\n".join(header)


def radiation_tensors_from_model(col):
    """Build vertical-frame JKQ [depth, K, Q+2] from SI model columns."""
    unknown = sorted(name for name in col if name.startswith("Jbar") and name not in J_COLUMNS)
    if unknown:
        raise ValueError(f"unsupported radiation tensor columns: {', '.join(unknown)}")
    n = len(col["height_km"])
    values = {}
    for name in J_COLUMNS:
        if name not in col:
            if name in ("Jbar00", "Jbar20"):
                raise ValueError(f"model must contain {name}")
            values[name] = np.zeros(n)
            continue
        value = np.asarray(col[name])
        if value.shape != (n,) or not np.isrealobj(value) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be real and finite with shape ({n},)")
        values[name] = value
    jkq = np.zeros((n, 3, 5), complex)
    jkq[:, 0, 2] = values["Jbar00"] * W_TO_CGS
    jkq[:, 2, 2] = values["Jbar20"] * W_TO_CGS
    for q in (1, 2):
        positive = (values[f"Jbar2{q}_re"] + 1j * values[f"Jbar2{q}_im"]) * W_TO_CGS
        jkq[:, 2, q + 2] = positive
        jkq[:, 2, 2 - q] = (-1) ** q * np.conj(positive)
    return jkq


def header_value(header, pattern, override, name):
    if override is not None:
        return override
    match = re.search(pattern, header)
    if match is None:
        raise ValueError(f"{name} not found in the header; pass it on the command line")
    return float(match.group(1))


def cumulative_tau(chi, dz):
    """Trapezoid-rule optical depth [depth, frequency] from the top for opacity chi [cm^-1]."""
    tau = np.zeros_like(chi)
    tau[1:] = np.cumsum(0.5 * (chi[:-1] + chi[1:]) * dz[:, None], axis=0)
    return tau


def continuum_from_tau500(col):
    """Continuum optical depth below the top node and opacity at the nodes, from the file's tau500.

    The depth is tau500 - tau500[0] (zero at the top node); the
    material above the top node is not included. chi_c = -d tau500 / dz [cm^-1] is
    evaluated as tau500 |d ln tau500 / dz| with a non-uniform central difference.
    """
    tau500 = 10.0 ** col["log_tau500"]
    if np.any(np.diff(tau500) <= 0):
        raise ValueError("log_tau500 must increase strictly with depth")
    z = col["height_km"] * 1e5
    return tau500 - tau500[0], -tau500 * np.gradient(np.log(tau500), z)


def denoise(y):
    """Zero U where it is pure rounding noise (an axisymmetric field has U = 0 for these rays)."""
    y = y.copy()
    scale = np.max(np.abs(y[1]))
    if scale > 0:
        y[2, np.abs(y[2]) < 1e-10 * scale] = 0.0
    return y


def stokes_percent(y):
    """Keep I and return Q/I, U/I in percent; mask non-positive intensities."""
    y = denoise(y)
    if not np.all(np.isfinite(y)):
        raise ValueError("Stokes values must be finite")
    invalid = y[0] <= 0
    if np.any(invalid):
        warnings.warn(f"Q/I and U/I undefined at {np.count_nonzero(invalid)} "
                      "points with non-positive I; those points are omitted",
                      RuntimeWarning, stacklevel=2)
    result = y.copy()
    result[1:] = np.nan
    np.divide(100 * y[1:], y[0], out=result[1:], where=~invalid)
    return result


def radiation_anisotropy(col):
    """Fractional anisotropy of the radiation tensors supplied in the model file."""
    if np.any(col["Jbar00"] <= 0):
        raise ValueError("Jbar00 must be positive to calculate anisotropy")
    return np.sqrt(2) * col["Jbar20"] / col["Jbar00"]


def plot_anisotropy(x, anisotropy, xlabel, path, logx=False):
    """Plot the prescribed file-moment anisotropy on a height or optical-depth axis."""
    keep = x > 0 if logx else np.ones(len(x), bool)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot(x[keep], anisotropy[keep])
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$w=\sqrt{2}\,\bar{J}^2_0/\bar{J}^0_0$")
    ax.set_title("Prescribed radiation anisotropy")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def panels(x, curves, xlabel, names, title, path, logx):
    keep = x > 0 if logx else np.ones(len(x), bool)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for i, (ax, name) in enumerate(zip(axes, names)):
        values = []
        for label, color, y in curves:
            ax.plot(x[keep], y[i][keep], color=color, label=label)
            values.append(y[i][keep])
        values = np.concatenate(values)
        values = values[np.isfinite(values)]
        peak = np.max(np.abs(values)) if values.size else 0.0
        if not values.size:
            ax.text(0.5, 0.5, "No defined values", transform=ax.transAxes, ha="center")
        elif peak == 0:
            ax.text(0.5, 0.5, "0 (axisymmetric field)", transform=ax.transAxes, ha="center")
        elif np.all(values > 0):
            ax.set_yscale("log")
        elif np.any(values < 0):
            ax.set_yscale("symlog", linthresh=1e-2 * peak)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spectral_coordinate(nu, line, axis):
    """Return sorted physical spectral coordinates, their order, and axis label."""
    absolute_nu = line.nu0 + np.asarray(nu)
    if not np.all(np.isfinite(absolute_nu)) or np.any(absolute_nu <= 0):
        raise ValueError("line-center frequency plus offsets must be finite and positive")
    if axis == "frequency":
        coordinate, xlabel = absolute_nu / 1e9, "Frequency [GHz]"
    elif axis == "wavelength":
        coordinate, xlabel = C_LIGHT / absolute_nu * 1e7, "Wavelength [nm]"
    else:
        raise ValueError("spectral axis must be 'frequency' or 'wavelength'")
    order = np.argsort(coordinate)
    return coordinate[order], order, xlabel


def format_spectral_axis(ax):
    """Keep absolute spectral coordinates readable on narrow panels."""
    lower, upper = ax.get_xlim()
    step = 10.0 ** np.ceil(np.log10(upper - lower))
    offset = np.floor((lower + upper) / (2 * step)) * step
    ax.ticklabel_format(axis="x", style="plain", useOffset=offset)
    ax.tick_params(axis="x", labelrotation=0)


def plot_spectrum_frequency(nu, stokes, scale, unit, mu, line, outdir, label_prefix=""):
    """Plot emergent I and fractional Q/I, U/I vs frequency for each viewing angle.

    Parameters
    ----------
    nu : array
        Frequency offsets from line center [Hz]
    stokes : array
        Stokes array [depth, IQU, frequency, mu, phi]; depth 0 is the top
    scale : float
        Conversion factor to units
    unit : str
        Unit label
    mu : array
        Cosine of viewing angles
    line : LineModel
        Line model with the line-center frequency nu0 [Hz]
    outdir : str
        Output directory
    label_prefix : str
        Prefix for filenames
    """
    os.makedirs(outdir, exist_ok=True)
    coordinate, order, xlabel = spectral_coordinate(nu, line, "frequency")

    for k, mu_val in enumerate(mu):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
        values = stokes_percent(scale * stokes[0, :3, :, k, 0])

        for i, (ax, label) in enumerate(zip(axes, ['I', 'Q/I', 'U/I'])):
            ax.plot(coordinate, values[i, order], color=f"C{k}")
            ax.set_xlabel(xlabel)
            format_spectral_axis(ax)
            ax.set_ylabel(f"{label} [{unit if i == 0 else '%'}]")
            ax.set_title(f"{label} spectrum at μ={mu_val:.2f}")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Emergent spectra at μ={mu_val:.2f}")
        filename = f"spectrum_freq_mu_{mu_val:.2f}.png"
        if label_prefix:
            filename = f"{label_prefix}_{filename}"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)


def plot_source_comparison(nu, stokes, reference, scale, unit, mu, line, outdir, prefix,
                           axis="frequency"):
    """Compare population and epsilon/J sources with the same opacity."""
    coordinate, order, xlabel = spectral_coordinate(nu, line, axis)
    for k, mu_val in enumerate(mu):
        population = denoise(scale * stokes[0, :3, :, k, 0])
        baseline = denoise(scale * reference[0, :3, :, k, 0])
        for normalized in (False, True):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
            arrays = (stokes_percent(population), stokes_percent(baseline)) if normalized else (population, baseline)
            labels = ("I", "Q/I", "U/I") if normalized else ("I", "Q", "U")
            for i, (ax, label) in enumerate(zip(axes, labels)):
                for values, name in zip(arrays, ("Population source", "Two-level ε/J source")):
                    ax.plot(coordinate, values[i, order], label=name)
                ax.set_xlabel(xlabel)
                format_spectral_axis(ax)
                ax.set_ylabel(f"{label} [{unit if not normalized or i == 0 else '%'}]")
                ax.grid(alpha=0.3)
            axes[0].legend()
            fig.suptitle(f"{prefix}: source comparison, μ={mu_val:.2f}")
            suffix = "percent" if normalized else "absolute"
            fig.savefig(os.path.join(outdir, f"{prefix}_source_comparison_{suffix}_{axis}_mu_{mu_val:.2f}.png"), dpi=150)
            plt.close(fig)
        p = stokes_percent(population)
        b = stokes_percent(baseline)
        delta_i = np.full_like(b[0], np.nan)
        np.divide(100 * (p[0] - b[0]), b[0], out=delta_i, where=b[0] > 0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
        for ax, values, label in zip(
            axes, (delta_i, p[1] - b[1], p[2] - b[2]),
            ("I difference / baseline [%]", "Q/I difference [percentage points]",
             "U/I difference [percentage points]"),
        ):
            ax.plot(coordinate, values[order])
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel(xlabel)
            format_spectral_axis(ax)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        fig.suptitle(f"{prefix}: population minus ε/J source, μ={mu_val:.2f}")
        fig.savefig(os.path.join(outdir, f"{prefix}_source_difference_{axis}_mu_{mu_val:.2f}.png"), dpi=150)
        plt.close(fig)


def plot_spectrum_normalized(nu, stokes, scale, unit, mu, line, outdir, label_prefix="",
                             axis="frequency"):
    """Plot normalized Q/I and U/I spectra for all viewing angles.

    Parameters
    ----------
    nu : array
        Frequency offsets from line center [Hz]
    stokes : array
        Stokes array [depth, IQU, frequency, mu, phi]; depth 0 is the top
    scale : float
        Conversion factor to units
    unit : str
        Unit label
    mu : array
        Cosine of viewing angles
    line : LineModel
        Line model with the line-center frequency nu0 [Hz]
    outdir : str
        Output directory
    label_prefix : str
        Prefix for filenames
    """
    os.makedirs(outdir, exist_ok=True)
    coordinate, order, xlabel = spectral_coordinate(nu, line, axis)

    for k, mu_val in enumerate(mu):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

        values = stokes_percent(scale * stokes[0, :3, :, k, 0])

        for ax, data, label in zip(axes, values[1:], ['Q/I', 'U/I']):
            ax.plot(coordinate, data[order], color=f"C{k}")
            ax.set_xlabel(xlabel)
            format_spectral_axis(ax)
            ax.set_ylabel(f"{label} [%]")
            ax.set_title(f"{label} spectrum at μ={mu_val:.2f}")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Emergent normalized polarization at μ={mu_val:.2f}")
        filename = f"spectrum_normalized_{axis}_mu_{mu_val:.2f}.png"
        if label_prefix:
            filename = f"{label_prefix}_{filename}"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)


def plot_spectrum_wavelength(nu, stokes, scale, unit, mu, line, outdir, label_prefix=""):
    """Plot emergent I and fractional Q/I, U/I vs wavelength for each viewing angle.

    Parameters
    ----------
    nu : array
        Frequency offsets from line center [Hz]
    stokes : array
        Stokes array [depth, IQU, frequency, mu, phi]; depth 0 is the top
    scale : float
        Conversion factor to units
    unit : str
        Unit label
    mu : array
        Cosine of viewing angles
    line : LineModel
        Line model with the line-center frequency nu0 [Hz]
    outdir : str
        Output directory
    label_prefix : str
        Prefix for filenames
    """
    os.makedirs(outdir, exist_ok=True)

    lam_sorted, sort_idx, xlabel = spectral_coordinate(nu, line, "wavelength")

    for k, mu_val in enumerate(mu):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
        values = stokes_percent(scale * stokes[0, :3, :, k, 0])

        for i, (ax, label) in enumerate(zip(axes, ['I', 'Q/I', 'U/I'])):
            ax.plot(lam_sorted, values[i, sort_idx], color=f"C{k}")
            ax.set_xlabel(xlabel)
            format_spectral_axis(ax)
            ax.set_ylabel(f"{label} [{unit if i == 0 else '%'}]")
            ax.set_title(f"{label} spectrum at μ={mu_val:.2f}")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Emergent spectra (wavelength) at μ={mu_val:.2f}")
        filename = f"spectrum_wavelength_mu_{mu_val:.2f}.png"
        if label_prefix:
            filename = f"{label_prefix}_{filename}"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)


def doppler_width_label(dnu_ref, line):
    """Reference frequency width and its small-offset wavelength equivalent."""
    dlambda_nm = line.lambda0 * dnu_ref / line.nu0 * 1e7
    return f"Reference Doppler width: {dlambda_nm:.5g} nm, {dnu_ref / 1e9:.5g} GHz"


def plot_spectrum_doppler(nu, stokes, scale, unit, mu, dnu_ref, line, outdir, label_prefix=""):
    """Plot emergent I and fractional Q/I, U/I vs Doppler-width frequency offset.

    Parameters
    ----------
    nu : array
        Frequency offsets from line center [Hz]
    stokes : array
        Stokes array [depth, IQU, frequency, mu, phi]; depth 0 is the top
    scale : float
        Conversion factor to units
    unit : str
        Unit label
    mu : array
        Cosine of viewing angles
    dnu_ref : float
        Reference Doppler width [Hz]
    line : LineModel
        Line model with nu0 (line center frequency)
    outdir : str
        Output directory
    label_prefix : str
        Prefix for filenames
    """
    os.makedirs(outdir, exist_ok=True)

    delta_nu = nu / dnu_ref

    for k, mu_val in enumerate(mu):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
        values = stokes_percent(scale * stokes[0, :3, :, k, 0])

        for i, (ax, label) in enumerate(zip(axes, ['I', 'Q/I', 'U/I'])):
            ax.plot(delta_nu, values[i], color=f"C{k}")
            ax.set_xlabel("Frequency offset [Doppler widths]")
            ax.set_ylabel(f"{label} [{unit if i == 0 else '%'}]")
            ax.set_title(f"{label} spectrum at μ={mu_val:.2f}")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Emergent spectra (Doppler) at μ={mu_val:.2f}")
        filename = f"spectrum_doppler_mu_{mu_val:.2f}.png"
        if label_prefix:
            filename = f"{label_prefix}_{filename}"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)


def plot_all_mu_spectra(nu, stokes, scale, unit, mu, line, outdir, label_prefix="",
                        axis="frequency"):
    """Plot I, Q/I, U/I spectra with all viewing angles overlaid.

    Parameters
    ----------
    nu : array
        Frequency offsets from line center [Hz]
    stokes : array
        Stokes array [depth, IQU, frequency, mu, phi]; depth 0 is the top
    scale : float
        Conversion factor to units
    unit : str
        Unit label
    mu : array
        Cosine of viewing angles
    line : LineModel
        Line model with the line-center frequency nu0 [Hz]
    outdir : str
        Output directory
    label_prefix : str
        Prefix for filenames
    """
    os.makedirs(outdir, exist_ok=True)
    coordinate, order, xlabel = spectral_coordinate(nu, line, axis)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    spectra = [stokes_percent(scale * stokes[0, :3, :, k, 0]) for k in range(len(mu))]

    for i, (ax, label) in enumerate(zip(axes, ['I', 'Q/I', 'U/I'])):
        for k, mu_val in enumerate(mu):
            ax.plot(coordinate, spectra[k][i, order],
                   label=f"μ={mu_val:.2f}", color=f"C{k}")
        ax.set_xlabel(xlabel)
        format_spectral_axis(ax)
        ax.set_ylabel(f"{label} [{unit if i == 0 else '%'}]")
        ax.set_title(f"{label} spectra")
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

    fig.suptitle("Emergent spectra for different viewing angles")
    filename = f"spectrum_all_mu_{axis}.png"
    if label_prefix:
        filename = f"{label_prefix}_{filename}"
    fig.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close(fig)


def plot_all_mu_spectra_doppler(nu, stokes, scale, unit, mu, dnu_ref, line, outdir, label_prefix=""):
    """Plot I, Q/I, U/I spectra with all viewing angles overlaid (Doppler axis).

    Parameters
    ----------
    nu : array
        Frequency offsets from line center [Hz]
    stokes : array
        Stokes array [depth, IQU, frequency, mu, phi]; depth 0 is the top
    scale : float
        Conversion factor to units
    unit : str
        Unit label
    mu : array
        Cosine of viewing angles
    dnu_ref : float
        Reference Doppler width [Hz]
    line : LineModel
        Line model with nu0 (line center frequency)
    outdir : str
        Output directory
    label_prefix : str
        Prefix for filenames
    """
    os.makedirs(outdir, exist_ok=True)

    delta_nu = nu / dnu_ref

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    spectra = [stokes_percent(scale * stokes[0, :3, :, k, 0]) for k in range(len(mu))]

    for i, (ax, label) in enumerate(zip(axes, ['I', 'Q/I', 'U/I'])):
        for k, mu_val in enumerate(mu):
            ax.plot(delta_nu, spectra[k][i],
                   label=f"μ={mu_val:.2f}", color=f"C{k}")
        ax.set_xlabel("Frequency offset [Doppler widths]")
        ax.set_ylabel(f"{label} [{unit if i == 0 else '%'}]")
        ax.set_title(f"{label} spectra")
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

    fig.suptitle("Emergent spectra for different viewing angles (Doppler)")
    filename = "spectrum_all_mu_doppler.png"
    if label_prefix:
        filename = f"{label_prefix}_{filename}"
    fig.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close(fig)


DEFAULT_MODEL = "sr4607_falc_Jbar_anisotropy.txt"


def resolve_model(path):
    """Use path as given, else look next to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (path, os.path.join(here, path)):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"model file {path!r} not found in the current directory or in {here}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                   help=f"model atmosphere text file (default: {DEFAULT_MODEL})")
    p.add_argument("--outdir", default=OUT)
    p.add_argument("--mu", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.33, 0.5, 0.75, 1.0])
    p.add_argument("--offsets", type=float, nargs="+", default=[0.0, 3.0, 6.0],
                   help="frequency offsets of the Stokes plots, in minimum local Doppler widths")
    p.add_argument("--extent", type=float, default=12.0,
                   help="half-width of the frequency grid, in minimum local Doppler widths")
    p.add_argument("--n-nu", type=int, default=241, help="number of frequency points (odd)")
    p.add_argument("--gamma-deg", type=float, default=90.0, help="Stokes reference angle (0: radial)")
    p.add_argument("--solver", choices=("sc", "fs"), default="sc",
                   help="sc: parabolic short characteristics; fs: exact linear-source propagation")
    p.add_argument("--spectral-axis", choices=("frequency", "wavelength", "both"), default="both",
                   help="spectral plot x-axis: absolute frequency, wavelength, or both (default)")
    p.add_argument("--units", choices=tuple(UNITS), default="si",
                   help="units of S and I in the plots: si (W m^-2 Hz^-1 sr^-1, default, as in the file)\n"
                        "or cgs (erg s^-1 cm^-2 Hz^-1 sr^-1)")
    p.add_argument("--bottom-I", type=float, default=None,
                   help="incoming bottom I in the units chosen with --units;\n"
                        "default is the Planck function at the bottom temperature")
    p.add_argument("--wavelength-nm", type=float, default=None, help="overrides the header")
    p.add_argument("--A-ul", type=float, default=None, help="overrides the header")
    p.add_argument("--gamma-rad", type=float, default=None, help="overrides the header")
    p.add_argument("--mass", type=float, default=87.62, help="atomic mass [amu] (Sr)")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def build_grid(col, line, v_turb, mu, nu, w_nu, chi_c, gamma, use_populations=False):
    """Grid carrying the file's J^K_Q, with emissivity and opacity filled for continuum chi_c [cm^-1]."""
    n = len(col["height_km"])
    atm = Atmosphere1D(z=col["height_km"] * 1e5, T=col["T_K"], v_z=np.zeros(n),
                       B=np.zeros((n, 3)), v_turb=v_turb, chi_c=chi_c)
    state = LineState(atm, line, col["n_lower"] * M3_TO_CM3, col["C_ul"], np.zeros(n),
                      a=col["a_damp"],
                      n_u=col["n_upper"] * M3_TO_CM3 if use_populations else None)
    # J^K_Q is supplied, so the angular weights are not used
    grid = Grid(atm, line, state, mu, np.full(len(mu), 2.0 / len(mu)),
                np.zeros(1), np.full(1, 2 * np.pi), nu, w_nu).allocate()
    grid.set_JKQ(radiation_tensors_from_model(col))
    fill_grid(grid, gamma=gamma)
    return grid


def transfer(source, tau, mu, line, T_bottom, bottom_I, solver):
    """I, Q, U [depth, 3, frequency, mu, 1]: dark top, unpolarized bottom (Planck by default)."""
    from formal_solver import solve_short_characteristics, solve_unpolarized_transfer

    shape = (3, source.shape[2], len(mu), 1)
    top, bottom = np.zeros(shape), np.zeros(shape)
    bottom[0] = planck(line.nu0, T_bottom) if bottom_I is None else bottom_I
    solve = solve_short_characteristics if solver == "sc" else solve_unpolarized_transfer
    return solve(source, tau, mu, top, bottom)


def main():
    args = parse_args()
    if args.n_nu < 3 or args.n_nu % 2 == 0:
        raise ValueError("--n-nu must be odd and at least 3")
    mu = np.array(args.mu)
    if np.any(mu <= 0) or np.any(mu > 1):
        raise ValueError("--mu values must be in (0, 1]")

    col, header = read_model(resolve_model(args.model))
    if "n_upper" not in col or "log_tau_line" not in col:
        raise ValueError("the population comparison requires n_upper and log_tau_line in the model file")
    lam0_nm = header_value(header, r"lambda0\s*=\s*([0-9.eE+-]+)\s*nm", args.wavelength_nm, "lambda0")
    a_ul = header_value(header, r"A_ul\s*=\s*([0-9.eE+-]+)", args.A_ul, "A_ul")
    gamma_rad = header_value(header, r"Gamma_rad\s*=\s*([0-9.eE+-]+)", args.gamma_rad, "Gamma_rad")
    lam0 = lam0_nm * 1e-7  # cm
    n = len(col["height_km"])

    # a = (Gamma_rad + Q_elast) lambda0 / (4 pi v_D) fixes v_D, hence the microturbulence
    q_elast = col.get("Q_elast", np.zeros(n))
    v_d2 = ((gamma_rad + q_elast) * lam0 / (4 * np.pi * col["a_damp"])) ** 2
    v_turb2 = v_d2 - 2 * K_B * col["T_K"] / (args.mass * AMU)
    if np.any(v_turb2 < -1e-3 * v_d2):
        raise ValueError("a_damp implies a Doppler width below the thermal one; check --mass")
    v_turb = np.sqrt(np.maximum(v_turb2, 0.0))

    line = LineModel(C_LIGHT / lam0, a_ul, 1.0, 1.0, 1, 0, args.mass)
    dnu_ref = line.doppler_width(col["T_K"], v_turb).min()
    nu = np.linspace(-args.extent * dnu_ref, args.extent * dnu_ref, args.n_nu)
    w_nu = np.full(args.n_nu, nu[1] - nu[0])
    w_nu[[0, -1]] *= 0.5
    gamma = np.deg2rad(args.gamma_deg)
    scale, unit = UNITS[args.units]
    bottom_I = None if args.bottom_I is None else args.bottom_I / scale  # internal units are cgs
    centre = int(np.argmin(np.abs(nu)))

    # line only
    grid = build_grid(col, line, v_turb, mu, nu, w_nu, np.zeros(n), gamma,
                      use_populations=True)
    atm, state, jkq = grid.atm, grid.line_state, grid.JKQ
    j = line.calc_j(grid, gamma=gamma)
    eta_i = grid.eta[:, 0]
    if np.any(eta_i <= 0):
        raise ValueError("line opacity is not positive everywhere; widen or shrink --extent")
    source = j / eta_i[:, None]
    tau = cumulative_tau(eta_i[:, :, 0, 0], -np.diff(atm.z))
    stokes = transfer(source, tau, mu, line, atm.T[-1], bottom_I, args.solver)
    baseline_grid = build_grid(col, line, v_turb, mu, nu, w_nu, np.zeros(n), gamma)
    baseline_source = baseline_grid.emis[:, :3] / baseline_grid.eta[:, 0, None]
    stokes_baseline = transfer(baseline_source, tau, mu, line, atm.T[-1], bottom_I, args.solver)

    # line + continuum: the file's tau500 is the continuum optical depth
    tau_cont, chi_c = continuum_from_tau500(col)
    grid_c = build_grid(col, line, v_turb, mu, nu, w_nu, chi_c, gamma,
                        use_populations=True)
    eta_total = grid_c.eta[:, 0]
    source_c = grid_c.emis[:, :3] / eta_total[:, None]
    tau_c = tau + tau_cont[:, None]
    stokes_c = transfer(source_c, tau_c, mu, line, atm.T[-1], bottom_I, args.solver)
    cc = chi_c[:, None, None, None]
    thermal = planck(line.nu0, atm.T)[:, None, None, None]
    baseline_source_c = baseline_source * eta_i[:, None] / eta_total[:, None]
    baseline_source_c[:, 0] += cc * thermal / eta_total
    stokes_c_baseline = transfer(baseline_source_c, tau_c, mu, line, atm.T[-1],
                                 bottom_I, args.solver)
    # checks against the file and against the source-function relations
    print(f"line: lambda0 = {lam0_nm:.4f} nm, A_ul = {a_ul:.4e} s^-1, "
          f"v_turb = {v_turb.min() / 1e5:.2f}..{v_turb.max() / 1e5:.2f} km/s (from a_damp)")
    print(f"epsilon: max |eps_pipeline/eps_file - 1| = "
          f"{np.max(np.abs(state.eps / col['epsilon'] - 1)):.2e}")
    tau_file = 10.0 ** col["log_tau_line"]
    tau_file = tau_file - tau_file[0]
    positive = tau_file > 0
    ratio = tau_c[positive, centre] / tau_file[positive]
    print(f"log_tau_line: reconstructed total tau / file tau (both relative to top): range "
          f"{ratio.min():.3f} .. {ratio.max():.3f}, median {np.median(ratio):.3f}")
    print(f"continuum: tau {tau_cont[1]:.2e} .. {tau_cont[-1]:.2e}, "
          f"chi_c {chi_c.min():.2e} .. {chi_c.max():.2e} cm^-1")
    if "w" in col:
        print(f"w: max |w - sqrt(2) J20/J00| = "
              f"{np.max(np.abs(col['w'] - np.sqrt(2) * col['Jbar20'] / col['Jbar00'])):.2e}")
    if len(mu) >= 2:
        # S_I(mu) = a + b (3 mu^2 - 1) with a = (1 - eps) J00 + eps B
        design = np.c_[np.ones(len(mu)), 3 * mu**2 - 1]
        s_mu = baseline_source[:, 0, centre, :, 0].T
        coef = np.linalg.lstsq(design, s_mu, rcond=None)[0]
        s_i = (1 - state.eps) * jkq[:, 0, 2].real + state.eps * planck(line.nu0, atm.T)
        print(f"epsilon/J S_I (line only): mu dependence follows a + b(3mu^2-1) to "
              f"{np.max(np.abs(design @ coef - s_mu) / s_mu):.1e}; "
              f"a = (1-eps)J00 + eps B to {np.max(np.abs(coef[0] / s_i - 1)):.1e}")
    print(f"S_I (line only) frequency spread: "
          f"{np.max(np.abs(source[:, 0] / source[:, 0, centre:centre + 1] - 1)):.2e}")
    expected_i = (j[:, 0] + cc * thermal) / (eta_i + cc)
    expected_q = j[:, 1] / (eta_i + cc)
    print(f"S (line + continuum) = (j + chi_c B)/(eta + chi_c) for I and j_Q/(eta + chi_c) for Q: "
          f"{np.max(np.abs(source_c[:, 0] / expected_i - 1)):.1e}, "
          f"{np.max(np.abs(source_c[:, 1] - expected_q) / np.abs(expected_q).max()):.1e}")
    print(f"max |S_U| / max |S_Q| = {np.abs(source[:, 2]).max() / np.abs(source[:, 1]).max():.2e}")
    print(f"minimum I [{args.units}]: line only {scale * stokes[:, 0].min():.3e}, "
          f"line + continuum {scale * stokes_c[:, 0].min():.3e} (should be >= 0)")

    os.makedirs(args.outdir, exist_ok=True)
    spectral_axes = ("frequency", "wavelength") if args.spectral_axis == "both" else (args.spectral_axis,)
    for axis in spectral_axes:
        plot_source_comparison(nu, stokes, stokes_baseline, scale, unit, mu, line,
                               args.outdir, "line", axis=axis)
        plot_source_comparison(nu, stokes_c, stokes_c_baseline, scale, unit, mu, line,
                               args.outdir, "continuum", axis=axis)
    height = col["height_km"]
    s_names = [rf"${s}$ [{unit}]" for s in ("S_I", "S_Q", "S_U")]
    stokes_names = [rf"$I$ [{unit}]", r"$Q/I$ [%]", r"$U/I$ [%]"]

    def rays(field, f, stokes_ratios=False):
        convert = stokes_percent if stokes_ratios else denoise
        return [(f"mu={m:.2f}", f"C{k}", convert(scale * field[:, :, f, k, 0].T))
                for k, m in enumerate(mu)]

    def offset_tag(offset):
        return f"{offset:g}_doppler_widths"

    def title(what, f):
        return f"{what} at {nu[f] / 1e9:+.2f} GHz from line centre"

    anisotropy = radiation_anisotropy(col)
    plot_anisotropy(height, anisotropy, "height [km]",
                    os.path.join(args.outdir, "anisotropy_vs_z.png"))
    plot_anisotropy(tau[:, centre], anisotropy, "line-only optical depth at line centre",
                    os.path.join(args.outdir, "anisotropy_vs_tau.png"), logx=True)
    plot_anisotropy(tau_c[:, centre], anisotropy, "line + continuum optical depth at line centre",
                    os.path.join(args.outdir, "anisotropy_continuum_vs_tau.png"), logx=True)

    # line only: S does not depend on frequency
    label = "line-only optical depth at line centre"
    panels(tau[:, centre], rays(source, centre), label, s_names, "Source functions (line only)",
           os.path.join(args.outdir, "source_vs_tau.png"), True)
    panels(height, rays(source, centre), "height [km]", s_names, "Source functions (line only)",
           os.path.join(args.outdir, "source_vs_z.png"), False)
    for offset in args.offsets:
        f = int(np.argmin(np.abs(nu - offset * dnu_ref)))
        panels(tau[:, centre], rays(stokes, f, stokes_ratios=True), label, stokes_names,
               title("Stokes solution (line only)", f),
               os.path.join(args.outdir, f"stokes_vs_tau_{offset_tag(offset)}.png"), True)

    # line + continuum: S depends on frequency through chi_c / eta
    label = "line + continuum optical depth at line centre"
    for offset in args.offsets:
        f = int(np.argmin(np.abs(nu - offset * dnu_ref)))
        tag = offset_tag(offset)
        name = title("Source functions (line + continuum)", f)
        panels(tau_c[:, centre], rays(source_c, f), label, s_names, name,
               os.path.join(args.outdir, f"source_continuum_vs_tau_{tag}.png"), True)
        panels(height, rays(source_c, f), "height [km]", s_names, name,
               os.path.join(args.outdir, f"source_continuum_vs_z_{tag}.png"), False)
        panels(tau_c[:, centre], rays(stokes_c, f, stokes_ratios=True), label, stokes_names,
               title("Stokes solution (line + continuum)", f),
               os.path.join(args.outdir, f"stokes_continuum_vs_tau_{tag}.png"), True)

    for axis in spectral_axes:
        for prefix, spectrum in (("line", stokes), ("continuum", stokes_c)):
            if axis == "frequency":
                plot_spectrum_frequency(nu, spectrum, scale, unit, mu, line, args.outdir,
                                        label_prefix=prefix)
            else:
                plot_spectrum_wavelength(nu, spectrum, scale, unit, mu, line, args.outdir,
                                         label_prefix=prefix)
            plot_spectrum_normalized(nu, spectrum, scale, unit, mu, line, args.outdir,
                                     label_prefix=prefix, axis=axis)
            plot_all_mu_spectra(nu, spectrum, scale, unit, mu, line, args.outdir,
                                label_prefix=prefix, axis=axis)

    print(f"plots written to {args.outdir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
