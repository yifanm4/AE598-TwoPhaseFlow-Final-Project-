import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Simple shell elasticity effects on RP (contrast-agent-like coating)

rho = 1.0
R0 = 1.0
p_g0 = 100.0
kappa = 1.4
p0 = 100.0
mu = 0.0
sigma = 0.0

Pa = 50.0          # keep drive mild to highlight stiffness shift
f = 0.2
omega = 2 * np.pi * f

chi_values = [0.0, 0.5, 1.0, 2.0]  # shell elasticity (nondimensional)


def rp_rhs_factory(chi):
    def p_bubble(R):
        R = max(R, 1e-8)
        return p_g0 * (R0 / R) ** (3 * kappa)

    def rhs(t, y):
        R, Rdot = y
        R = max(R, 1e-8)
        p_inf = p0 + Pa * np.sin(omega * t)
        delta_p = p_bubble(R) - p_inf
        shell_term = 4 * chi * (R - R0) / (R0 * R)  # linear shell tension
        surf = 2 * sigma / R
        Rddot = (delta_p - surf - shell_term) / (rho * R) - 1.5 * (Rdot ** 2) / R
        return [Rdot, Rddot]

    return rhs


periods = 6
t_end = periods / f
t_eval = np.linspace(0, t_end, 1500)
y0 = [R0, 0.0]

plt.figure(figsize=(6.5, 5))
for chi in chi_values:
    rhs = rp_rhs_factory(chi)
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    plt.plot(sol.t / (1 / f), sol.y[0] / R0, label=f"chi={chi}")

plt.xlabel("t / period")
plt.ylabel("R / R0")
plt.title("Shell elasticity effect on oscillations (RP + shell term)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("study_shell_effects.png", dpi=300)
plt.show()
