# Bubble Dynamics Numerical Studies

This folder contains small, self-contained Python studies of spherical bubble dynamics using Rayleigh–Plesset (RP) and Keller–Miksis (KM) equations, plus supporting plots and reference PDFs.

## Requirements
- Python 3.10+ recommended
- `numpy`, `scipy`, `matplotlib`
- Install with `pip install -r requirements.txt` (or `pip install numpy scipy matplotlib` if no requirements file exists).

## Key scripts and outputs
- `final-bubble-evolution-basilisk.py` — nondimensional RP/KM inviscid and viscous cases; writes `inviscid_bubble_radius_vs_time.png`, `viscous_bubble_radius_vs_time.png`, `bubble_entropy_vs_time.png`, `bubble_interfaces.png`.
- `final-bubble-evolution-class-slide.py` — water microbubble at 1 MHz, 150 kPa drive; compares RP vs KM; writes `bubble_dynamics.png`.
- `pinf_bubble_study.py` — RP/KM far-field pressure sweep; writes `pinf_bubble_radius_vs_time.png`.
- `study_pressure_frequency.py` — RP heatmap of peak `R/R0` vs drive amplitude/frequency; writes `study_pressure_frequency.png`.
- `study_ambient_pressure.py` — static `p_inf` sweep for collapse time and minimum radius; writes `study_ambient_pressure.png`.
- `study_damping.py` — viscosity/surface tension sweep on peak amplitude; writes `study_damping.png`.
- `study_compressibility.py` — KM peak amplitude vs sound speed with RP baseline; writes `study_compressibility.png`.
- `study_thermal_gaslaw.py` — gas-law sensitivity via `kappa` sweep; writes `study_thermal_gaslaw.png`.
- `study_mass_transfer_vapor.py` — vapor pressure term sweep; writes `study_mass_transfer_vapor.png`.
- `study_shell_effects.py` — simple shell elasticity term; writes `study_shell_effects.png`.
- `study_bifurcation_map.py` — Poincaré samples vs drive amplitude (bifurcation-style); writes `study_bifurcation_map.png`.
- `study_energy_pressure.py` — radiated-pressure proxy from RP trajectory; writes `study_energy_pressure.png`.
- `study_bubble_cloud.py` — ensemble of different initial sizes under common drive; writes `study_bubble_cloud.png`.

Reference PDFs (papers/notes) are included for background and are not needed to run the scripts.

## How to run
From this folder, execute any script directly, e.g.:
```bash
python study_pressure_frequency.py
```
Each script saves its figure(s) alongside the code and displays them interactively (`matplotlib` default).
