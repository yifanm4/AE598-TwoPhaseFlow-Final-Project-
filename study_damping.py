import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Damping sweep: viscosity and surface tension effects on peak amplitude (RP)

rho = 1.0
R0 = 1.0
p_g0 = 100.0
kappa = 1.4
p0 = 100.0

Pa = 150.0
f = 0.2
omega = 2 * np.pi * f

mu_values = np.logspace(-4, -1, 8)   # viscous sweep
sigma_values = [0.0, 0.01, 0.05, 0.1]  # surface tension sweep


def p_bubble(R):
    R = max(R, 1e-8)
    return p_g0 * (R0 / R) ** (3 * kappa)


def rp_rhs(t, y, mu, sigma):
    R, Rdot = y
    R = max(R, 1e-8)
    p_inf = p0 + Pa * np.sin(omega * t)
    delta_p = p_bubble(R) - p_inf
    visc = 4 * mu * Rdot / R
    surf = 2 * sigma / R
    Rddot = (delta_p - visc - surf) / (rho * R) - 1.5 * (Rdot ** 2) / R
    return [Rdot, Rddot]


def run(mu, sigma):
    periods = 8
    t_end = periods / f
    t_eval = np.linspace(0, t_end, 1800)
    y0 = [R0, 0.0]
    sol = solve_ivp(lambda t, y: rp_rhs(t, y, mu, sigma),
                    (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    return sol.y[0].max() / R0


plt.figure(figsize=(6.5, 5))
for sigma in sigma_values:
    peaks = [run(mu, sigma) for mu in mu_values]
    plt.semilogx(mu_values, peaks, marker="o", label=f"sigma={sigma}")

plt.xlabel("Viscosity mu")
plt.ylabel("Peak R/R0")
plt.title("Damping impact on oscillation amplitude (RP)")
plt.grid(True, which="both", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("study_damping.png", dpi=300)
plt.show()
