import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Compressibility sensitivity using Keller–Miksis vs RP

rho = 1.0
R0 = 1.0
p_g0 = 100.0
kappa = 1.4
p0 = 100.0
mu = 0.0
sigma = 0.0

Pa = 150.0
f = 0.2
omega = 2 * np.pi * f

c_values = [5.0, 10.0, 20.0, 50.0, 100.0]  # nondimensional sound speed


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


def km_rhs(t, y, c):
    R, Rdot = y
    R = max(R, 1e-8)
    p_inf = p0 + Pa * np.sin(omega * t)
    dp_inf = Pa * omega * np.cos(omega * t)

    pB = p_bubble(R)
    dpB = -3 * kappa * (Rdot / R) * pB

    delta_p = pB - p_inf
    d_delta = dpB - dp_inf

    denom = (1 - Rdot / c) * R
    if abs(denom) < 1e-12:
        denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12

    A = (1 + Rdot / c) * delta_p / rho
    B = (R / (rho * c)) * d_delta
    C = 1.5 * (1 - Rdot / (3 * c)) * Rdot ** 2
    Rddot = (A + B - C) / denom
    return [Rdot, Rddot]


def peak_radius(rhs):
    periods = 8
    t_end = periods / f
    t_eval = np.linspace(0, t_end, 1800)
    y0 = [R0, 0.0]
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    return sol.y[0].max() / R0


rp_peak = peak_radius(rp_rhs)
km_peaks = [peak_radius(lambda t, y, c=c: km_rhs(t, y, c)) for c in c_values]

plt.figure(figsize=(6.5, 4.5))
plt.axhline(rp_peak, color="gray", linestyle=":", label="RP peak")
plt.plot(c_values, km_peaks, marker="o", label="KM peak")
plt.xlabel("Sound speed c (nondimensional)")
plt.ylabel("Peak R/R0")
plt.title("Compressibility sensitivity (KM vs RP)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("study_compressibility.png", dpi=300)
plt.show()
