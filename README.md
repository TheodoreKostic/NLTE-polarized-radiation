# NLTE-polarized-radiation
Non-LTE radiative transfer of polarized radiation in presence of magnetic field 

# Aim
Using "A Multilevel Radiative Transfer Program for Modeling Scattering Line Polarization and the Hanle Effect in Stellar Atmospheres", Sainz and Bueno (2003)
and Javier Trujillo Bueno and Rafael Manso Sainz 1999 ApJ <b>516</b> 436 as references, we want to create a code that can be used to solve radiative transfer problem
for polarized radiation field due to presence of magnetic field.

# Flow
<ol>
  <li>Compute source functions S<sub>I</sub> and S<sub>Q</sub>. They are a function of $\mu$. For starters, we shall use Doppler line profile.</li>
  <li>How does microturbulent (intensity only) magnetic field impact the source functions?</li>
  <li>Full magnetic field vector -> S<sub>I</sub>, S<sub>Q</sub>, <b>S<sub>U</sub></b>. In general, they are coupled.</li>
</ol>
