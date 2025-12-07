import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Radiated pressure proxy: compute far-field pulse from RP trajectory

rho = 1.0
R0 = 1.0
p_g0 = 100.0
kappa = 1.4
p0 = 100.0
mu = 0.0
sigma = 0.0

Pa = 200.0
f = 0.2
omega = 2 * np.pi * f
r_obs = 100.0 * R0  # observation distance


def p_bubble(R):
    R = max(R, 1e-8)
    return p_g0 * (R0 / R) ** (3 * kappa)


def rp_rhs(t, y):
    R, Rdot = y
    R = max(R, 1e-8)
    p_inf = p0 + Pa * np.sin(omega * t)
    delta_p = p_bubble(R) - p_inf
    Rddot = delta_p / (rho * R) - 1.5 * (Rdot ** 2) / R
    return [Rdot, Rddot]


periods = 6
t_end = periods / f
t_eval = np.linspace(0, t_end, 2000)
y0 = [R0, 0.0]
sol = solve_ivp(rp_rhs, (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)

R_series = sol.y[0]
Rdot_series = sol.y[1]
Rddot_series = []
for R, Rdot, t in zip(R_series, Rdot_series, sol.t):
    p_inf = p0 + Pa * np.sin(omega * t)
    delta_p = p_bubble(R) - p_inf
    Rddot = delta_p / (rho * R) - 1.5 * (Rdot ** 2) / R
    Rddot_series.append(Rddot)

Rddot_series = np.array(Rddot_series)

# Radiated pressure from Rayleigh approximation: p_sc = rho (R * Rddot + 2 Rdot^2) * R / r_obs
p_sc = rho * (R_series * Rddot_series + 2 * Rdot_series ** 2) * R_series / r_obs

plt.figure(figsize=(7, 4))
plt.plot(sol.t / (1 / f), p_sc, label="p_scattered (scaled)")
plt.xlabel("t / period")
plt.ylabel("Pressure (arb.)")
plt.title("Radiated pressure waveform at r/R0=100 (RP)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("study_energy_pressure.png", dpi=300)
plt.show()
