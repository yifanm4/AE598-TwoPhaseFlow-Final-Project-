import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --------------------------------------
# Baseline nondimensional parameters
# --------------------------------------
rho = 1.0           # Density
R0 = 1.0            # Initial radius
p_g0 = 100.0        # Initial gas pressure
gamma = 1.4         # Polytropic index

# Pressure sweep for the study (far-field values)
P_INF_VALUES = [50, 100, 1000, 10000]

# --------------------------------------
# Helper functions
# --------------------------------------
def p_gas(R: float) -> float:
    """Adiabatic gas pressure inside the bubble."""
    return p_g0 * (R0 / R) ** (3 * gamma)


def model_rhs(t, y, *, p_inf: float, c_liquid: float, model: str) -> list[float]:
    """Right-hand sides for Rayleigh–Plesset (RP) and Keller–Miksis (KM)."""
    R, Rdot = y
    R = max(R, 1e-8)  # Prevent singularities if radius collapses

    pg = p_gas(R)
    dpg_dt = -3 * gamma * (Rdot / R) * pg

    delta_p = pg - p_inf         # Driving pressure difference
    visc_term = 0.0              # Inviscid study

    if model == "RP":
        Rddot = (delta_p - visc_term) / (rho * R) - 1.5 * (Rdot ** 2) / R
    elif model == "KM":
        c = c_liquid
        A = (1 + Rdot / c) / rho * (delta_p - visc_term)
        B = (R / (rho * c)) * dpg_dt  # dp_inf/dt = 0 (constant far-field pressure)
        C = 1.5 * (1 - Rdot / (3 * c)) * Rdot ** 2
        denom = (1 - Rdot / c) * R
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
        Rddot = (A + B - C) / denom
    else:
        raise ValueError(f"Unknown model: {model}")

    return [Rdot, Rddot]


def simulate_case(p_inf: float, t_factor: float = 2.5):
    """
    Run RP and KM for a given far-field pressure.
    Returns normalized time and radius arrays for plotting.
    """
    # Pressure gap sets the characteristic time; floor avoids division by zero near equilibrium.
    delta_p_mag = max(abs(p_inf - p_g0), 1.0)
    tR = 0.915 * R0 * np.sqrt(rho / delta_p_mag)
    t_end = t_factor * tR
    c_liquid = 10.0 * np.sqrt(delta_p_mag / rho)

    t_eval = np.linspace(0.0, t_end, 800)
    y0 = [R0, 0.0]

    sol_rp = solve_ivp(
        lambda t, y: model_rhs(t, y, p_inf=p_inf, c_liquid=c_liquid, model="RP"),
        (0.0, t_end), y0, t_eval=t_eval, rtol=1e-9, atol=1e-12
    )
    sol_km = solve_ivp(
        lambda t, y: model_rhs(t, y, p_inf=p_inf, c_liquid=c_liquid, model="KM"),
        (0.0, t_end), y0, t_eval=t_eval, rtol=1e-9, atol=1e-12
    )

    return {
        "p_inf": p_inf,
        "tR": tR,
        "t_norm": sol_rp.t / tR,
        "R_rp": sol_rp.y[0] / R0,
        "R_km": sol_km.y[0] / R0,
    }


# --------------------------------------
# Run the sweep and plot
# --------------------------------------
cases = [simulate_case(p) for p in P_INF_VALUES]

plt.figure(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(cases)))

for color, case in zip(colors, cases):
    label_base = f"p_inf={case['p_inf']/p_g0:.1f} pg0"
    plt.plot(case["t_norm"], case["R_rp"], color=color, linestyle="-", label=f"RP, {label_base}")
    plt.plot(case["t_norm"], case["R_km"], color=color, linestyle="--", label=f"KM, {label_base}")

plt.xlabel("t / tR")
plt.ylabel("R / R0")
plt.title("Effect of Far-Field Pressure on Bubble Radius (Inviscid)")
plt.grid(True, alpha=0.4)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("pinf_bubble_radius_vs_time.png", dpi=300)
plt.show()
