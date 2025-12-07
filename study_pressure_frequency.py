import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Pressure–frequency sweep for RP with sinusoidal forcing

rho = 1.0
R0 = 1.0
p_g0 = 100.0
kappa = 1.4
p0 = 100.0          # ambient around equilibrium bubble
mu = 0.0
sigma = 0.0

Pa_vals = np.linspace(25, 300, 15)      # drive amplitude sweep
f_vals = np.linspace(0.05, 0.8, 15)     # drive frequency sweep (nondimensional)


def p_bubble(R):
    R = max(R, 1e-8)
    return p_g0 * (R0 / R) ** (3 * kappa)


def rp_rhs(t, y, Pa, omega):
    R, Rdot = y
    R = max(R, 1e-8)
    p_inf = p0 + Pa * np.sin(omega * t)
    delta_p = p_bubble(R) - p_inf
    visc = 4 * mu * Rdot / R
    surf = 2 * sigma / R
    Rddot = (delta_p - visc - surf) / (rho * R) - 1.5 * (Rdot ** 2) / R
    return [Rdot, Rddot]


def run_case(Pa, f):
    omega = 2 * np.pi * f
    periods = 8
    t_end = periods / f
    t_eval = np.linspace(0, t_end, 2000)
    y0 = [R0, 0.0]
    sol = solve_ivp(lambda t, y: rp_rhs(t, y, Pa, omega),
                    (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    max_R = sol.y[0].max()
    return max_R / R0


heat = np.zeros((len(f_vals), len(Pa_vals)))
for i, f in enumerate(f_vals):
    for j, Pa in enumerate(Pa_vals):
        heat[i, j] = run_case(Pa, f)

# Plot heatmap of peak radius
plt.figure(figsize=(6.5, 5))
extent = [Pa_vals[0], Pa_vals[-1], f_vals[0], f_vals[-1]]
im = plt.imshow(heat, origin="lower", aspect="auto", extent=extent, cmap="magma")
plt.xlabel("Drive amplitude Pa")
plt.ylabel("Frequency (nondimensional)")
plt.title("Peak R/R0 vs drive amplitude & frequency (RP)")
plt.colorbar(im, label="Max R/R0")
plt.tight_layout()
plt.savefig("study_pressure_frequency.png", dpi=300)
plt.show()
