import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Static far-field pressure sweep: collapse time and minimum radius

rho = 1.0
R0 = 1.0
p_g0 = 100.0
kappa = 1.4
mu = 0.0
sigma = 0.0

p_inf_values = np.array([20, 50, 80, 100, 200, 500, 1000])


def p_bubble(R):
    R = max(R, 1e-8)
    return p_g0 * (R0 / R) ** (3 * kappa)


def rp_rhs(t, y, p_inf):
    R, Rdot = y
    R = max(R, 1e-8)
    delta_p = p_bubble(R) - p_inf
    visc = 4 * mu * Rdot / R
    surf = 2 * sigma / R
    Rddot = (delta_p - visc - surf) / (rho * R) - 1.5 * (Rdot ** 2) / R
    return [Rdot, Rddot]


collapse_times = []
min_radii = []

for p_inf in p_inf_values:
    delta_p = max(abs(p_inf - p_g0), 1.0)
    tR = 0.915 * R0 * np.sqrt(rho / delta_p)
    t_end = 3 * tR
    t_eval = np.linspace(0, t_end, 1200)
    y0 = [R0, 0.0]
    sol = solve_ivp(lambda t, y: rp_rhs(t, y, p_inf),
                    (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    R_series = sol.y[0]
    min_r = R_series.min() / R0
    # crude collapse time: time to minimum radius
    t_col = sol.t[np.argmin(R_series)] / tR
    collapse_times.append(t_col)
    min_radii.append(min_r)

# Plot
fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
ax1.plot(p_inf_values, min_radii, "o-", label="Min R/R0")
ax1.set_xlabel("p_inf (static)")
ax1.set_ylabel("Min R/R0")
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(p_inf_values, collapse_times, "s--", color="crimson", label="Collapse time (t/tR)")
ax2.set_ylabel("Collapse time (t/tR)")

fig.suptitle("Static far-field pressure: collapse response (RP)")
fig.tight_layout()
plt.savefig("study_ambient_pressure.png", dpi=300)
plt.show()
