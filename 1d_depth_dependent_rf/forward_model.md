# Fixed-population magnetic forward model

This guide follows the model-file workflow in
[`plot_physical_flow.py`](../Polarization-in-spectral-lines/SolveRTE/plot_physical_flow.py).
It calculates emergent Stokes spectra for prescribed atmospheric properties,
level populations and radiation-field tensors. It is the forward-model part of
the [response-function plan](flow.md); it does **not** calculate response
functions or iterate the radiation field to NLTE self-consistency.

## Why can U and V appear?

The prescribed $\bar J^K_Q$ describe the **incident, profile-averaged
radiation field** in the vertical frame. They are inputs to the
atomic-polarization calculation, not the outgoing Stokes parameters.
The supplied example file contains only $\bar J^0_0$ and $\bar J^2_0$.
Setting the other input tensor components to zero does not require the
emitted or emergent Stokes $U$ and $V$ to be zero:

1. **U:** $\bar J^2_0$ pumps upper-level alignment. An inclined magnetic field
   breaks axial symmetry and changes that alignment through the Hanle effect.
   The emitted linear polarization can then have a U component in the chosen
   Stokes reference frame. Zeeman-induced linear polarization and
   magneto-optical coupling during propagation can also contribute. Q and U
   depend on the chosen reference direction.
2. **V:** Zeeman splitting produces different profiles for circular
   polarization components. Their emission and absorption coefficients can
   produce V, including when $\bar J^2_0=0$. Magneto-optical coupling can also
   transfer polarization between Stokes components.

The field direction is fixed in the atmospheric frame; changing $\mu$ changes
the **viewing direction**, and therefore its angle to the field. The model
holds the original tensors fixed when B changes. In particular, it does not
recompute field-induced radiation tensors from the emergent polarization.

For example, with the supplied model, a vertical 5 G field, $\mu=0.5$,
$D^{(2)}=0$ and the default Stokes reference angle, a 31-point test gives a
peak $|U/I|$ of about 0.0012%. Setting only the magneto-optical coefficients
to zero for that same formal solution reduces U to numerical roundoff. Thus
even a vertical field can generate U **during propagation** in this
calculation; that is distinct from inclined-field Hanle rotation of the local
emission. This is a model calculation, not an observational benchmark.

## Ordered calculation

1. **Choose the model and calculation options.**

   `main()` parses the options and calls `read_model()` and `resolve_model()`
   from [`plot_model_file.py`](../Polarization-in-spectral-lines/SolveRTE/plot_model_file.py).
   The default input is
   [`sr4607_falc_Jbar_anisotropy.txt`](../Polarization-in-spectral-lines/SolveRTE/sr4607_falc_Jbar_anisotropy.txt),
   a static, unmagnetized FALC/Lightweaver model. Rows run from the top
   downward. Its header describes the quantities and notes that the
   collisional rates are preliminary.

   The columns used here include height, temperature, $n_l$, $n_u$,
   $\bar J^0_0$, $\bar J^2_0$, $C_{ul}$ and the Voigt damping parameter.
   `log_tau_line` is **total vertical line-center optical depth**, including
   continuum, not line-only optical depth. `log_tau500` is used for a printed
   continuum comparison, not as the transfer optical-depth scale in this
   workflow.

   The file gives populations in m$^{-3}$ and radiation tensors in
   W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$. The code converts them to cm$^{-3}$ and
   erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$, respectively.

2. **Construct the line and Doppler widths.**

   `model_parameters()` reads $\lambda_0$, $A_{ul}$ and the radiative
   broadening rate from the header. It constructs `LineModel` from
   [`atmosphere.py`](../Polarization-in-spectral-lines/SolveRTE/atmosphere.py)
   for the $J_l=0\rightarrow J_u=1$ normal triplet, with upper and lower
   Landé factors set to 1. The input file's `g_l=1, g_u=3` are **statistical
   weights**, not those Landé factors.

   The function infers microturbulent velocity from the tabulated damping
   parameter and broadening rates. `LineModel.doppler_width()` then evaluates
   the depth-dependent thermal-plus-microturbulent Doppler width:

   $$
   \Delta\nu_D(z)={\nu_0\over c}
      \sqrt{{2kT(z)\over m}+v_\mathrm{turb}^2(z)},\qquad
   \Delta\lambda_D(z)={\lambda_0\over\nu_0}\Delta\nu_D(z).
   $$

   The tabulated `Q_elast` participates in the broadening calculation; it is
   **not** substituted for the separate depolarization rate $D^{(2)}$.

3. **Construct the wavelength and ray grids.**

   `wavelength_grid()` creates an increasing, equally spaced vacuum-wavelength
   grid. By default it spans 12 reference Doppler widths on each side of line
   center with 241 points. The reference width is the smallest local
   $\Delta\lambda_D$. `--lambda-range-nm` can instead set the endpoints.
   The code also calculates $\nu=c/\lambda$ and frequency offsets
   $\nu-\nu_0$.

   `--mu` selects outward rays, with $\mu=\cos\theta$ measured from the
   outward vertical. The default rays are 0.01, 0.05, 0.1, 0.33, 0.5, 0.75 and 1.0.
   This workflow sets the observer azimuth to zero: the rays
   lie in the x-z plane. $\mu=1$ is vertical; $\mu=0.5$ is 60 degrees from
   vertical. The constant Cartesian field is specified independently by
   `--b-field BX BY BZ` in gauss. Positive z points outward.

4. **Infer the static continuum from line-center optical depth.**

   Unless `--line-only` is selected, `continuum_from_line_center()` converts
   the tabulated $\log\tau_\mathrm{line}$ to $\tau$ and reconstructs the
   nodal total line-center opacity using the tabulated trapezoidal integration
   convention and top half-cell. It calls `build_grid()` for the **zero-field,
   static, population-based** line-center opacity and subtracts that opacity
   from the reconstructed total:

   $$
   \chi_c(z)\approx\chi_\mathrm{total}(\lambda_0,z)
                      -\chi_\mathrm{line}(\lambda_0,z;B=0).
   $$

   The remainder is treated as gray, unpolarized, thermal continuum. This
   reconstruction is held fixed when the field or velocity is changed; it is
   not an independently supplied continuum opacity. Inverting tabulated
   optical depths can amplify rounding errors, and negative inferred
   continuua are rejected. The printed comparison with a `tau500`-based
   estimate is a comparison of approximations, not an accuracy guarantee.

5. **Create the atmosphere, line state and prescribed tensors.**

   `physical_flow()` creates `Atmosphere1D`, `LineState` and `Grid` from
   [`atmosphere.py`](../Polarization-in-spectral-lines/SolveRTE/atmosphere.py).
   It supplies depth-dependent $T$, $n_l$, $n_u$, $C_{ul}$ and damping, the
   inferred $\chi_c$, and the selected field, velocity and $D^{(2)}$.
   At the command line the field, vertical velocity and $D^{(2)}$ are
   constant with depth. The Python `physical_flow()` API also accepts
   depth-dependent B, unresolved-field strength and $D^{(2)}$ arrays.

   `radiation_tensors_from_model()` in `plot_model_file.py` builds the
   `[depth, K, Q+2]` array passed to `Grid.set_JKQ()`. `Jbar00` and `Jbar20`
   are required. The four additional columns below are optional, real-valued
   numbers in the same units as `Jbar00` and `Jbar20`:

   | Column | Vertical-frame component |
   | --- | --- |
   | `Jbar21_re` | $\operatorname{Re}\bar J^2_1$ |
   | `Jbar21_im` | $\operatorname{Im}\bar J^2_1$ |
   | `Jbar22_re` | $\operatorname{Re}\bar J^2_2$ |
   | `Jbar22_im` | $\operatorname{Im}\bar J^2_2$ |

   Add any desired column names to the existing `# height_km ...` header
   line and add the corresponding values to **every** data row in the same
   order. Each omitted real or imaginary column is independently zero.
   The code constructs
   $\bar J^2_1=\texttt{Jbar21_re}+i\,\texttt{Jbar21_im}$ and similarly for
   $\bar J^2_2$, then fills the negative-Q components using
   $\bar J^2_{-Q}=(-1)^Q(\bar J^2_Q)^*$. Unsupported `Jbar` column names
   are rejected, rather than silently ignored.

   Thus the input comprises six independent **real numbers**: two real
   $Q=0$ components and the real and imaginary parts for $Q=1,2$. Rank-1
   radiation moments are not included. The supplied example file has no
   optional columns, so its original zero values are unchanged. These
   moments and the populations are held fixed, not updated from the
   calculated Stokes profiles. Non-axisymmetric prescribed moments can be
   used, but the atmosphere and transfer calculation remain 1D.

6. **Calculate the reduced wavelength at every depth and ray.**

   `Atmosphere1D.projected_velocity()` computes $\mathbf v\cdot\hat{\mathbf n}$.
   The script's positive vertical velocity is outward, so for a ray directed
   toward the observer the nonrelativistic coordinate is

   $$
   x(z,\lambda,\mu)=
   {\lambda-\lambda_0[1-v_\mathrm{proj}(z,\mu)/c]\over\Delta\lambda_D(z)}.
   $$

   `physical_flow()` returns `x` with shape `[depth, wavelength, mu, 1]`.
   The final axis is the single observer azimuth. The opacity routine
   independently evaluates the same wavelength-coordinate profile at each
   depth and ray.

7. **Rotate the radiation anisotropy and calculate the Hanle response.**

   `fill_grid()` and `_grid_coefficients()` in
   [`grid_opacity.py`](../Polarization-in-spectral-lines/SolveRTE/grid_opacity.py)
   evaluate the coefficients at each depth. `Atmosphere1D.magnetic_angles()`
   converts B to a strength, inclination from +z and azimuth from +x toward
   +y. `magnetic_frame_geometry()` also expresses each viewing ray in the
   magnetic frame.

   `rho2_magnetic()` uses a Wigner rotation to express the rank-2 radiation
   tensor in that frame. Even if only $\bar J^2_0$ was supplied in the
   vertical frame, a tilted-frame representation can have multiple
   $\bar J^2_Q$ components. It then applies the two-level Hanle and
   collisional response:

   $$
   R^2_Q={(\mathsf D^\dagger \bar J^2)_Q
     \over 1+(C_{ul}+D^{(2)})/A_{ul}+iQH_u},\qquad
   H_u={2\pi\,\nu_\mathrm{Larmor}\,g_u B\over A_{ul}}.
   $$

   Here $\nu_\mathrm{Larmor}=1.3996\times10^6$ Hz/G. The variable called
   `rho2` in the code is a **source-unit response**, not the physical
   population-normalized density matrix on its own. Inclination, Hanle
   precession and collisions affect the resulting alignment and coherences.
   The command-line `--d2` takes s$^{-1}$; equivalently
   $\delta^{(2)}=D^{(2)}/A_{ul}$. Its default is zero.

8. **Compute Zeeman profiles and local coefficients.**

   `point_emission_and_propagation()` calls `phi_table()` and
   `phi_table_disp()` for each depth and ray. These use the complex Voigt
   function from [`line_profiles.py`](../Polarization-in-spectral-lines/SolveRTE/line_profiles.py),
   shifted for the three Zeeman components, and Wigner coefficients to
   construct generalized absorption and dispersion profiles. The angular
   tensors come from `tensor_basis_angles()` in
   [`grid_tensors.py`](../Polarization-in-spectral-lines/SolveRTE/grid_tensors.py).

   In the population branch used by this workflow:

   - The line propagation amplitude uses the population difference
     $n_l-(g_l/g_u)n_u$, with $g_l/g_u=1/3$ here, for the scalar
     stimulated-emission correction.
   - Isotropic spontaneous emission is set by $n_uA_{ul}$. For the
     **zero-field isotropic line-only** limit, the scalar source is
     $S_L=n_u A_{ul}/(n_l B_{lu}-n_u B_{ul})$, using the code's Einstein
     and intensity convention.
   - The aligned-emission contribution is pumped by $n_l\bar J^2_Q$ and
     modified by the response in step 7. It does not use a thermal
     stimulated factor.
   - The supplied $\bar J^0_0$ is stored, but does **not** directly determine
     isotropic emission in this population-based branch; $n_u$ does. The
     tabulated `epsilon` is not used to blend this branch's source with
     $\bar J^0_0$ and the Planck function.

   The routine rotates Q/U-related quantities back to the vertical-frame
   Stokes reference and adds $\chi_c$ to $\eta_I$ and
   $\chi_c B_\nu(T)$ to $j_I$. Continuum emission uses the Planck function
   at each grid frequency; the population line source uses $\nu_0$.

   `fill_grid()` stores `grid.emis` with shape
   `[depth, 4, wavelength, mu, 1]`, ordered $(j_I,j_Q,j_U,j_V)$, and
   `grid.eta` with shape `[depth, 7, wavelength, mu, 1]`, ordered
   $(\eta_I,\eta_Q,\eta_U,\eta_V,\rho_Q,\rho_U,\rho_V)$.
   The last three $\rho$ values are magneto-optical propagation coefficients,
   not the atomic `rho2` of step 7.

9. **Optionally average over an unresolved field.**

   If `--b-unres` is nonzero, `total_field_samples()` constructs fields
   $\mathbf B_\mathrm{resolved}+B_\mathrm{unres}\hat{\mathbf n}$ with
   isotropically distributed directions. `point_emission_and_propagation_unresolved()`
   averages the local emission and propagation coefficients over those
   directions before solving transfer. This is a microstructured-field
   approximation, not an average of independently emergent spectra.
   `--n-theta` and `--n-chi` control angular quadrature; check convergence,
   particularly for strong comparable resolved and unresolved fields.

10. **Build optical depths and solve the transfer equation.**

    The returned diagnostic `tau` integrates the computed $\eta_I$ over
    vertical depth for each wavelength and ray using `cumulative_tau()`,
    including a top half-cell offset. With a magnetic field it can depend on
    viewing direction. For a ray, a cell's optical-depth increment is divided
    by $|\mu|$.

    With `--solver delo`, `solve_delo()` and `delo_polarized_ray()` in
    [`formal_solver.py`](../Polarization-in-spectral-lines/SolveRTE/formal_solver.py)
    solve the coupled equation along each ray:

    $$
    {d\mathbf I\over ds}=\mathbf j-\mathsf K\mathbf I,\qquad
    \mathbf I=(I,Q,U,V)^T,
    $$

    $$
    \mathsf K=
    \begin{pmatrix}
    \eta_I&\eta_Q&\eta_U&\eta_V\\
    \eta_Q&\eta_I&\rho_V&-\rho_U\\
    \eta_U&-\rho_V&\eta_I&\rho_Q\\
    \eta_V&\rho_U&-\rho_Q&\eta_I
    \end{pmatrix}.
    $$

    The solver uses trapezoidal $\eta_I$ cell optical depths and a DELO-linear
    interpolation of the effective source. It sweeps upward from an
    unpolarized Planck-function bottom boundary to the top grid point.
    No transfer is solved through the added top half-cell.

    `physical_flow()` also returns `source=j[:3]/eta_I` for inspection.
    In polarized transfer this is **not** generally $\mathsf K^{-1}\mathbf j$,
    nor are the individual Stokes equations independent. The default
    `--solver sc` instead solves separate I/Q/U scalar short-characteristic
    equations and cannot be used with a nonzero resolved or unresolved field.

11. **Extract and plot the emergent spectra.**

    `plot_spectra()` takes the Stokes vector at top depth index 0. For DELO
    it plots I and percentages $100Q/I$, $100U/I$ and $100V/I$ for each
    selected $\mu$. `--spectral-axis wavelength`, `frequency`, or `both`
    (the default) selects the horizontal spectral coordinate. Frequency is
    absolute $\nu$ in GHz; wavelength is in nm. The wavelength grid is still
    used for the calculation and frequency is its conversion $\nu=c/\lambda$;
    neither plot uses the depth-dependent reduced coordinate $x$.
    Spectral ticks show deviations from the additive offset beside the axis.
    Add that offset to a tick label to obtain the absolute frequency or
    wavelength.
    For display only, $|100V/I|<10^{-15}$ percent is plotted as zero; the
    computed Stokes vector is not altered.
    Frequency plots use $I_\nu$ per Hz;
    wavelength plots convert to $I_\lambda$ per nm. Filenames and titles
    include resolved field strength and angles; azimuth is undefined for a
    vertical field and both angles are undefined for a zero field.

    The returned `stokes` array has shape
    `[depth, 4, wavelength, mu, 1]` for DELO. For scalar short
    characteristics it has three Stokes components; the plot displays V as
    zero in that case, rather than calculating polarized V.

    The separate `plot_model_file.py` script accepts the same
    `--spectral-axis` option, including for comparison and all-$\mu$
    spectra. It solves on a frequency grid and converts it to wavelength
    for wavelength-axis plots. Its intensity remains $I_\nu$ per Hz on
    either axis. Plots against height or optical depth keep those axes.
    Existing files in an output directory are not removed, so use a new
    directory if you do not want older Doppler-axis plots mixed with the new
    output.

## Example

From the repository root, a 5 G outward vertical field and
$D^{(2)}=0.1A_{ul}=2.0099\times10^7$ s$^{-1}$ for the supplied line:

```bash
python Polarization-in-spectral-lines/SolveRTE/plot_physical_flow.py \
  --solver delo --b-field 0 0 5 --d2 2.0099e7 \
  --outdir Polarization-in-spectral-lines/SolveRTE/plots_magnetic
```

Changing the field alone keeps the input moments, populations and inferred
static continuum fixed. Varying `--d2` changes the assumed elastic
depolarization; the file's broadening rate is not a measurement of D2.

## Scope and limitations

- The input populations are from a multilevel NLTE calculation, but the
  magnetic alignment is a **two-level approximation**, not a self-consistent
  polarized multilevel solution. Other transitions' alignment pumping and
  relaxation are omitted.
- Stimulated emission is included as a population correction to opacity, but
  its effects on upper-level alignment and polarized propagation are neglected.
  The input moments also remain fixed when Zeeman splitting or velocity would
  change profile-weighted pumping.
- The gray thermal continuum treats all inferred background opacity,
  including any scattering contribution, as thermal emission. The
  line-center-opacity inversion is sensitive to rounding.
- The input file flags preliminary, uncertain collisional rates. Results are
  therefore model-dependent; agreement with a plot is not validation of the
  approximation or of the rates.
- The formal solver synthesizes spectra for fixed coefficients. It does not
  update populations, radiation moments or the continuum, and does not yet
  propagate magnetic response functions.
