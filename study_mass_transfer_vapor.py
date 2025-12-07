import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Vapor / mass-transfer proxy: vary vapor pressure term p_v in RP

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

p_v_values = [0.0, 10.0, 25.0, 50.0]  # increasing vapor fraction


def rp_rhs_factory(p_v):
    def p_bubble(R):
        R = max(R, 1e-8)
        return p_v + p_g0 * (R0 / R) ** (3 * kappa)

    def rhs(t, y):
        R, Rdot = y
        R = max(R, 1e-8)
        p_inf = p0 + Pa * np.sin(omega * t)
        delta_p = p_bubble(R) - p_inf
        Rddot = delta_p / (rho * R) - 1.5 * (Rdot ** 2) / R
        return [Rdot, Rddot]

    return rhs


periods = 6
t_end = periods / f
t_eval = np.linspace(0, t_end, 1500)
y0 = [R0, 0.0]

plt.figure(figsize=(6.5, 5))
for p_v in p_v_values:
    rhs = rp_rhs_factory(p_v)
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    plt.plot(sol.t / (1 / f), sol.y[0] / R0, label=f"p_v={p_v}")

plt.xlabel("t / period")
plt.ylabel("R / R0")
plt.title("Vapor pressure effect on oscillations (RP)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("study_mass_transfer_vapor.png", dpi=300)
plt.show()
