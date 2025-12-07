import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Gas law / thermal model sensitivity via kappa sweep (RP)

rho = 1.0
R0 = 1.0
p_g0 = 100.0
p0 = 100.0
mu = 0.0
sigma = 0.0

Pa = 150.0
f = 0.2
omega = 2 * np.pi * f

kappa_values = [1.0, 1.1, 1.3, 1.4]  # isothermal -> adiabatic


def rp_rhs_factory(kappa):
    def p_bubble(R):
        R = max(R, 1e-8)
        return p_g0 * (R0 / R) ** (3 * kappa)

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
for kappa in kappa_values:
    rhs = rp_rhs_factory(kappa)
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-11)
    plt.plot(sol.t / (1 / f), sol.y[0] / R0, label=f"kappa={kappa}")

plt.xlabel("t / period")
plt.ylabel("R / R0")
plt.title("Effect of gas law (kappa) on bubble oscillations (RP)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("study_thermal_gaslaw.png", dpi=300)
plt.show()
