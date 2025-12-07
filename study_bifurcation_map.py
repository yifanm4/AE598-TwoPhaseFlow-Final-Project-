import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Bifurcation-style map: Poincaré samples vs drive amplitude (RP)

rho = 1.0
R0 = 1.0
p_g0 = 100.0
kappa = 1.4
p0 = 100.0
mu = 0.0
sigma = 0.0

f = 0.2
omega = 2 * np.pi * f
Pa_values = np.linspace(10, 300, 60)


def p_bubble(R):
    R = max(R, 1e-8)
    return p_g0 * (R0 / R) ** (3 * kappa)


def rp_rhs(t, y, Pa):
    R, Rdot = y
    R = max(R, 1e-8)
    p_inf = p0 + Pa * np.sin(omega * t)
    delta_p = p_bubble(R) - p_inf
    Rddot = delta_p / (rho * R) - 1.5 * (Rdot ** 2) / R
    return [Rdot, Rddot]


poincare_Pa = []
poincare_R = []

period = 1.0 / f

for Pa in Pa_values:
    t_end = 30 * period
    t_eval = np.linspace(0, t_end, 4000)
    y0 = [R0, 0.0]
    sol = solve_ivp(lambda t, y: rp_rhs(t, y, Pa),
                    (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    # sample once per period after transient
    sample_times = np.arange(10 * period, t_end, period)
    R_interp = np.interp(sample_times, sol.t, sol.y[0])
    poincare_Pa.extend([Pa] * len(R_interp))
    poincare_R.extend(R_interp / R0)

plt.figure(figsize=(6.5, 5))
plt.scatter(poincare_Pa, poincare_R, s=6, c=poincare_R, cmap="plasma", alpha=0.8)
plt.xlabel("Drive amplitude Pa")
plt.ylabel("Poincaré-sampled R/R0")
plt.title("Bifurcation-style map vs drive amplitude (RP)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("study_bifurcation_map.png", dpi=300)
plt.show()
