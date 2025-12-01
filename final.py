import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# Physical parameters (water)
# -----------------------------
rho = 1000.0       # kg/m^3, density
mu = 1.0e-3        # Pa·s, dynamic viscosity
sigma = 0.072      # N/m, surface tension
c = 1500.0         # m/s, sound speed

# -----------------------------
# Bubble / gas parameters
# -----------------------------
R0 = 2e-6          # m, initial radius = 2 microns
kappa = 1.4        # polytropic exponent (adiabatic)
p_v = 2.3e3        # Pa, vapor pressure
p0 = 1.0e5         # Pa, ambient pressure

# -----------------------------
# Pressure forcing: 150 kPa, 1 MHz
# -----------------------------
Pa = 150e3         # Pa, pressure amplitude (150 kPa)
f = 1e6            # Hz, frequency (1 MHz)
omega = 2.0 * np.pi * f

# Gas pressure at equilibrium at t = 0:
# p_B(0) - p_inf(0) - 2σ/R0 = 0  => p_g0 = p0 + 2σ/R0 - p_v
p_g0 = p0 + 2.0 * sigma / R0 - p_v

# -----------------------------
# Helper functions
# -----------------------------
def p_infinity(t):
    """Far-field pressure (sinusoidal)."""
    return p0 + Pa * np.sin(omega * t)

def dp_infinity_dt(t):
    """Time derivative of far-field pressure."""
    return Pa * omega * np.cos(omega * t)

def p_bubble(R):
    """Bubble internal pressure (gas + vapor) via polytropic gas law."""
    R = max(R, 1e-12)
    p_g = p_g0 * (R0 / R) ** (3.0 * kappa)
    return p_v + p_g

def dp_bubble_dt(R, Rdot):
    """Time derivative of bubble internal pressure."""
    R = max(R, 1e-12)
    p_g = p_g0 * (R0 / R) ** (3.0 * kappa)
    dp_g_dt = -3.0 * kappa * (Rdot / R) * p_g
    return dp_g_dt

# -----------------------------
# Rayleigh–Plesset RHS
# -----------------------------
def rp_rhs(t, y):
    """
    Right-hand side of the Rayleigh–Plesset equation.
    y = [R, Rdot]
    """
    R, Rdot = y
    R = max(R, 1e-12)

    p_inf = p_infinity(t)
    pB = p_bubble(R)

    # Rayleigh–Plesset:
    # rho (R Rddot + 3/2 Rdot^2) = pB - p_inf - 2σ/R - 4μ Rdot/R
    term_pressure = pB - p_inf
    term_sigma = 2.0 * sigma / R
    term_visc = 4.0 * mu * Rdot / R

    Rddot = (term_pressure - term_sigma - term_visc) / (rho * R) \
            - 1.5 * (Rdot ** 2) / R

    return [Rdot, Rddot]

# -----------------------------
# Keller–Miksis RHS
# -----------------------------
def km_rhs(t, y):
    """
    Right-hand side of the Keller–Miksis equation.
    y = [R, Rdot]
    """
    R, Rdot = y
    R = max(R, 1e-12)

    # Pressures and derivatives
    p_inf = p_infinity(t)
    dp_inf = dp_infinity_dt(t)
    pB = p_bubble(R)
    dpB = dp_bubble_dt(R, Rdot)

    delta_p = pB - p_inf
    d_delta_p = dpB - dp_inf

    term_sigma = 2.0 * sigma / R
    term_visc = 4.0 * mu * Rdot / R

    # Keller–Miksis:
    # (1 - Rdot/c) R Rddot + 3/2 (1 - Rdot/(3c)) Rdot^2
    # = (1/rho) (1 + Rdot/c) [delta_p - 2σ/R - 4μ Rdot/R]
    #   + (R/(rho c)) d(delta_p)/dt
    A = (1.0 / rho) * (1.0 + Rdot / c) * (delta_p - term_sigma - term_visc) \
        + (R / (rho * c)) * d_delta_p

    denom = (1.0 - Rdot / c) * R
    inertial_corr = 1.5 * (1.0 - Rdot / (3.0 * c)) * Rdot**2

    # Avoid singular denominator
    if abs(denom) < 1e-12:
        denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12

    Rddot = (A - inertial_corr) / denom
    return [Rdot, Rddot]

# -----------------------------
# Time integration: 0–3 microseconds
# -----------------------------
t_start = 0.0
t_end = 3e-6                # 3 microseconds
t_eval = np.linspace(t_start, t_end, 2000)

y0 = [R0, 0.0]              # initial radius, zero velocity

# Solve Rayleigh–Plesset
sol_rp = solve_ivp(
    rp_rhs, (t_start, t_end), y0,
    t_eval=t_eval, rtol=1e-9, atol=1e-12, method="RK45"
)

# Solve Keller–Miksis
sol_km = solve_ivp(
    km_rhs, (t_start, t_end), y0,
    t_eval=t_eval, rtol=1e-9, atol=1e-12, method="RK45"
)

# -----------------------------
# Plot R(t) in microns / microseconds
# -----------------------------
t_us = sol_rp.t * 1e6
R_rp_um = sol_rp.y[0] * 1e6
R_km_um = sol_km.y[0] * 1e6

plt.figure(figsize=(4, 4))
plt.plot(t_us, R_rp_um, label="Rayleigh–Plesset")
plt.plot(t_us, R_km_um, "--", label="Keller–Miksis")
plt.xlabel(r"$t\ [\mu s]$")
plt.ylabel(r"$R\ [\mu m]$")
plt.xlim(0, 3)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bubble_dynamics.png", dpi=300)
plt.show()
