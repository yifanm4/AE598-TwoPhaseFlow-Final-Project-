import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Simple ensemble study: distribution of bubble sizes under same drive (independent RP)

rho = 1.0
p_g0 = 100.0
kappa = 1.4
p0 = 100.0
mu = 0.0
sigma = 0.0

Pa = 150.0
f = 0.2
omega = 2 * np.pi * f

R0_values = np.linspace(0.5, 1.5, 7)  # different initial sizes


def rp_rhs_factory(R0):
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
t_common = np.linspace(0, t_end, 1500)

R_tracks = []
for R0 in R0_values:
    rhs = rp_rhs_factory(R0)
    y0 = [R0, 0.0]
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_common, rtol=1e-8, atol=1e-11)
    # If solver stops early, interpolate onto the common grid; pad with NaN beyond solution interval.
    R_interp = np.interp(t_common, sol.t, sol.y[0], left=np.nan, right=np.nan) / R0
    R_tracks.append(R_interp)

R_tracks = np.array(R_tracks)
mean_track = np.nanmean(R_tracks, axis=0)
min_track = np.nanmin(R_tracks, axis=0)
max_track = np.nanmax(R_tracks, axis=0)

plt.figure(figsize=(6.5, 5))
plt.fill_between(t_common / (1 / f), min_track, max_track, color="lightblue", alpha=0.5, label="min/max across sizes")
plt.plot(t_common / (1 / f), mean_track, color="navy", label="mean R/R0")
plt.title("Ensemble response of bubble size distribution (RP)")
plt.xlabel("t / period")
plt.ylabel("R / R0 (normalized per bubble)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("study_bubble_cloud.png", dpi=300)
plt.show()
