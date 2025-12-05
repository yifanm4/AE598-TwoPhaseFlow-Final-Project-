import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Circle

# ==========================================
# 1. Physical Parameters (from Basilisk src/test/collapse.c)
# ==========================================
# Units are non-dimensional
rho = 1.0           # Density
R0 = 1.0            # Initial radius
p_g0 = 100.0        # Initial gas pressure [cite: 27]
p_inf = 500.0       # Far-field pressure (5 * pg0) [cite: 28]
p_v = 0.0           # Vapor pressure (assumed negligible)
gamma = 1.4         # Polytropic index [cite: 128]
Reynolds = 10.0     # Reynolds number [cite: 120]

# Derived Parameters
# Viscosity derived from Reynolds number: mu = sqrt(p_inf - p_g0) / Re
mu_viscous = np.sqrt(p_inf - p_g0) / Reynolds
mu_inviscid = 0.0

# Rayleigh Collapse Time (tR) for normalization [cite: 30]
# tR = 0.915 * R0 * sqrt(rho / (p_inf - p_g0))
tR = 0.915 * R0 * np.sqrt(rho / (p_inf - p_g0))

# Sound speed for Keller-Miksis (Approximation for "slightly compressible")
# High enough to be stable, finite for KM physics
c_liquid = 10.0 * np.sqrt((p_inf - p_g0)/rho)

# ==========================================
# 2. Solver Functions (RP and KM)
# ==========================================
def p_gas(R):
    """Adiabatic gas law: P * V^gamma = constant."""
    return p_g0 * (R0 / R)**(3 * gamma)

def model_derivatives(t, y, mu_val, model_type='RP'):
    R, Rdot = y
    R = max(R, 1e-6) # Safety floor to prevent singularity

    # Pressures
    pg = p_gas(R)
    dpg_dt = -3 * gamma * (Rdot / R) * pg # Time derivative of gas pressure

    delta_p = pg - p_inf # Driving pressure difference (pv=0)
    d_delta_p_dt = dpg_dt # dp_inf/dt = 0 (constant far-field pressure)

    # Viscous term
    visc_term = 4.0 * mu_val * Rdot / R
    # Surface tension is 0 in this test case [cite: 7]

    if model_type == 'RP':
        # Rayleigh-Plesset Equation
        numerator = delta_p - visc_term
        Rddot = (numerator / (rho * R)) - 1.5 * (Rdot**2 / R)

    elif model_type == 'KM':
        # Keller-Miksis Equation
        c = c_liquid
        A = (1 + Rdot/c)/rho * (delta_p - visc_term)
        B = (R/(rho*c)) * d_delta_p_dt
        C = 1.5 * (1 - Rdot/(3*c)) * Rdot**2
        D = (1 - Rdot/c) * R

        Rddot = (A + B - C) / D

    return [Rdot, Rddot]

# ==========================================
# 3. Run Simulations
# ==========================================
t_span = [0, 2.5 * tR]
t_eval = np.linspace(0, 2.5 * tR, 1000)
y0 = [R0, 0.0] # Initial conditions: R=R0, Velocity=0

# Case A: Inviscid (mu=0)
sol_rp_inv = solve_ivp(lambda t, y: model_derivatives(t, y, mu_inviscid, 'RP'),
                       t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
sol_km_inv = solve_ivp(lambda t, y: model_derivatives(t, y, mu_inviscid, 'KM'),
                       t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

# Case B: Viscous (mu > 0)
sol_rp_vis = solve_ivp(lambda t, y: model_derivatives(t, y, mu_viscous, 'RP'),
                       t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
sol_km_vis = solve_ivp(lambda t, y: model_derivatives(t, y, mu_viscous, 'KM'),
                       t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

# ==========================================
# 4. Generate Individual Plots
# ==========================================

# --- Figure 1: Inviscid Radius vs Time ---
plt.figure(figsize=(6, 5))
plt.plot(sol_rp_inv.t/tR, sol_rp_inv.y[0]/R0, 'g-', label='Rayleigh-Plesset')
plt.plot(sol_km_inv.t/tR, sol_km_inv.y[0]/R0, 'b--', label='Keller-Miksis')
plt.title('Inviscid Fluid: Radius vs Time')
plt.xlabel('t / tR')
plt.ylabel('R / R0')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("inviscid_bubble_radius_vs_time.png", dpi=300)
plt.show()

# --- Figure 2: Viscous Radius vs Time ---
plt.figure(figsize=(6, 5))
plt.plot(sol_rp_vis.t/tR, sol_rp_vis.y[0]/R0, 'g-', label='Rayleigh-Plesset')
plt.plot(sol_km_vis.t/tR, sol_km_vis.y[0]/R0, 'y-', label='Keller-Miksis')
# Add faint inviscid KM line for reference (as done in PDF)
plt.plot(sol_km_inv.t/tR, sol_km_inv.y[0]/R0, 'b:', alpha=0.5, label='KM (Inviscid Ref)')
plt.title('Viscous Fluid: Radius vs Time')
plt.xlabel('t / tR')
plt.ylabel('R / R0')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("viscous_bubble_radius_vs_time.png", dpi=300)
plt.show()

# --- Figure 3: Entropy Errors (Theoretical) ---
plt.figure(figsize=(6, 5))
# Calculate p * V^gamma for the viscous KM solution
vol_vis = (4/3) * np.pi * sol_km_vis.y[0]**3
entropy_vis = p_gas(sol_km_vis.y[0]) * vol_vis**gamma

plt.plot(sol_km_vis.t/tR, entropy_vis, 'k-', label='Theoretical (ODE)')
plt.title('Entropy (pV^gamma) vs Time')
plt.xlabel('t / tR')
plt.ylabel('p V^gamma')
plt.ylim(min(entropy_vis)*0.99, max(entropy_vis)*1.01) # Zoom to show it's constant
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bubble_entropy_vs_time.png", dpi=300)
plt.show()

# --- Figure 4: Interfaces (Bubble Shape) ---
plt.figure(figsize=(6, 5))
ax = plt.gca()
ax.set_aspect('equal')
t_indices = np.linspace(0, len(sol_km_vis.t)-1, 15, dtype=int)

for i in t_indices:
    r_current = sol_km_vis.y[0][i] / R0
    # Draw concentric circles representing the bubble wall
    circle = Circle((0, 0), r_current, fill=False, color='purple', alpha=0.5)
    ax.add_patch(circle)

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_title('Interfaces (Bubble Shape Evolution)')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.savefig("bubble_interfaces.png", dpi=300)
plt.show()
