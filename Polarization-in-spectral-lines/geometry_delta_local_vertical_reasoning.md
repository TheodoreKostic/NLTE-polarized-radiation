# Geometry of δ, the Local Vertical, and the Line of Sight

## Context

In the Chapter 13 Hanle calculations, the question is how the angle δ should enter the geometry when the radiation tensors \(T^K_Q(i,\Omega)\) are evaluated.

The important distinction is between:

1. a coordinate system whose \(z\)-axis is the **local vertical** at the scattering point, and
2. a fixed/global coordinate system.

The current implementation uses the **local-vertical frame**.

## 1. The local vertical is already the z-axis

The LOS vector is constructed as

```python
def _los_vec(theta, chi):
    return np.array([
        np.sin(theta) * np.cos(chi),
        np.sin(theta) * np.sin(chi),
        np.cos(theta)
    ], dtype=float)
```

Here θ is the polar angle measured from the local vertical.

Therefore, when the coordinate system is defined with the local vertical as its z-axis, the local vertical itself does **not** need to be rotated by δ.

The quantity that changes is the direction of the line of sight relative to that local vertical.

## 2. Why θ_obs = π/2 − δ

If δ is the angle between the line of sight and the local horizontal direction, then the angle between the LOS and the local vertical is its complementary angle:

\[
\theta_{\rm obs} = \frac{\pi}{2}-\delta.
\]

Thus:

- δ = 0° gives θ_obs = 90°;
- δ = 30° gives θ_obs = 60°;
- δ = 90° gives θ_obs = 0°.

So using

```python
theta_obs = np.pi/2 - delta
```

is not necessarily an empirical trick. It is the natural conversion between two ways of describing the same LOS direction, provided δ is defined as the angle from the local horizontal.

## 3. Geometrical picture

For δ = 0°, the LOS is horizontal in the local frame:

```text
             local vertical
                  |
                  |
                  |
                  +-----------> LOS
```

Hence θ_obs = 90°.

For δ > 0°, the LOS tilts toward the local vertical:

```text
             local vertical
                  |
                  |\
                  | \
                  |  \ LOS
                  |   \
                  +---->
```

and therefore θ_obs decreases.

For δ = 90°, the LOS is parallel to the local vertical:

```text
             LOS
              ↑
              |
              |
              |
              |
              +
```

so θ_obs = 0°.

## 4. Why the vertical should not be rotated in the present implementation

The current frame transformation contains

```python
los_vert = _los_vec(theta_obs, chi_obs)
los_mag = _rotate_vert_to_mag(los_vert, theta_B, chi_B)
```

The name `los_vert` is important: it is the LOS expressed **in the local-vertical frame**.

The local vertical is already the z-axis of that frame. Consequently, there is no additional operation such as

```python
vertical_rotated = rotate_by_delta(vertical)
```

before `_rotate_vert_to_mag`.

The correct sequence is:

1. Define the LOS in the local-vertical frame.
2. Set its polar angle from the geometry:
   \[
   \theta_{\rm obs}=\pi/2-\delta.
   \]
3. Define the corresponding local Q/U reference direction.
4. Transform both the LOS and reference direction into the magnetic frame.
5. Compute the magnetic-frame angles and transported Q/U reference.

Schematic:

```text
       LOCAL-VERTICAL FRAME
              |
              | LOS defined by delta
              | theta_obs = pi/2 - delta
              v
          (los_vert)
              |
              | rotate by magnetic-field orientation
              v
        MAGNETIC FRAME
              |
              v
          (los_mag)
```

## 5. What would be different in a fixed/global frame?

If the starting coordinate system were a fixed global frame, then the local vertical at the scattering point would generally not coincide with the global z-axis.

Then an additional transformation would be required:

```text
global frame
     |
     | transform to local frame
     v
local-vertical frame
     |
     | transform to magnetic frame
     v
magnetic frame
```

In that case one could have

```python
los_local = R_global_to_local @ los_global
los_mag = R_local_to_mag @ los_local
```

But that is not what the present `_los_vec(theta, chi)` construction represents. It directly constructs the LOS in the local-vertical frame.

## 6. The role of h_R(δ)

There is a separate appearance of δ in the height/radius relation,

\[
h_R(\delta)
=
\frac{1+H_P}{\cos\delta}-1.
\]

This should not be interpreted as another rotation of the radiation-tensor coordinate system.

It describes the geometrical location of the scattering point and therefore changes the anisotropy of the incident radiation field.

The two appearances of δ have different meanings:

### A. In the anisotropy

\[
\delta \longrightarrow h_R(\delta)
\longrightarrow \text{anisotropy}.
\]

### B. In the LOS geometry

\[
\delta \longrightarrow
\theta_{\rm obs}=\frac{\pi}{2}-\delta
\longrightarrow T^K_Q(i,\Omega).
\]

They therefore describe two distinct physical effects and are not double-counting.

## 7. The δ = 90° limit

The expression

\[
h_R(\delta)
=
\frac{1+H_P}{\cos\delta}-1
\]

does diverge as δ → 90°, because cos δ → 0.

That divergence belongs to the **geometrical height relation**, not to the LOS tensor geometry.

At the same time,

\[
\theta_{\rm obs}
=
\frac{\pi}{2}-\delta
\]

has the perfectly regular limit

\[
\theta_{\rm obs}\rightarrow0.
\]

Thus there is no geometrical singularity in the radiation-tensor direction itself. The singularity in h_R simply says that, within the particular geometrical construction being used, reaching exactly δ = 90° corresponds to a scattering point at arbitrarily large height.

## 8. Why recovering the LL04 figures matters

An important numerical observation is that using

```python
theta_obs = np.pi/2 - delta
```

recovers the correct shape of the δ = 30° Hanle diagrams (Figs. 13.4 and 13.5), while the δ = 0° cases are also recovered.

That is significant evidence that the angular interpretation is consistent with the coordinate geometry.

It is stronger evidence than simply saying that the substitution happens to produce the desired curve: the same choice has a clear geometrical interpretation in a local-vertical frame.

Nevertheless, this is a consistency check rather than, by itself, a formal derivation of every detail of the LL04 geometry.

## 9. Practical conclusion for the current code

For the present local-vertical implementation, the appropriate construction is:

```python
theta_obs = np.pi/2 - delta
chi_obs = 0.0
gamma_obs = np.pi/2
```

followed by the existing local-to-magnetic-frame transformation:

```python
los_vert = _los_vec(theta_obs, chi_obs)
los_mag = _rotate_vert_to_mag(los_vert, theta_B, chi_B)
```

The local vertical should **not** be separately rotated by δ.

The conceptual picture is:

\[
\boxed{
\text{local vertical is fixed as the }z\text{-axis};
\quad
\delta\text{ determines the LOS angle within that frame.}
}
\]

The conclusion depends on the precise definition of δ in Fig. 13.1. If δ is defined relative to a different direction, then the complementary-angle relation must be reconsidered.
