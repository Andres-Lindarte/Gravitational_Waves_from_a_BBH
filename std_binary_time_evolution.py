from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

GAMMA_E = 0.5772156649015329 # Euler-Mascheroni constant

# Function to solve the system of ODEs
def taylort4_rhs(t, y, nu):
    x, phi = y
    
    # Post-Newtonian 3.5PN coefcicients (G=c=m=1)
    c1  = -(743.0 / 336.0) - (11.0 / 4.0) * nu
    c15 = 4.0 * np.pi
    c2  = (34103.0 / 18144.0) + (13661.0 / 2016.0) * nu + (59.0 / 18.0) * (nu**2)
    c25 = -((4159.0 / 672.0) + (189.0 / 8.0) * nu) * np.pi
    c3  = (
        (16447322263.0 / 139708800.0)
        - (1712.0 / 105.0) * GAMMA_E
        + (16.0 / 3.0) * (np.pi**2)
        - (56198689.0 / 217728.0) * nu
        + (451.0 / 48.0) * (np.pi**2) * nu
        + (541.0 / 896.0) * (nu**2)
        - (5605.0 / 2592.0) * (nu**3)
        - (856.0 / 105.0) * np.log(16.0 * x)
    )
    c35 = (-(4415.0 / 4032.0) + (358675.0 / 6048.0) * nu + (91495.0 / 1512.0) * (nu**2)) * np.pi

    # 3.5PN polynomial in x
    v = np.sqrt(x) # Precalculates x^(1/2) for efficiency
    poly = (
        1.0
        + c1 * x
        + c15 * x * v
        + c2 * (x**2)
        + c25 * (x**2) * v
        + c3 * (x**3)
        + c35 * (x**3) * v
    )

    dx_dt = (64.0 / 5.0) * nu * (x**5) * poly
    dphi_dt = x * v  # x^(3/2)

    return [dx_dt, dphi_dt]

# Event funtion to stop the integration
def ISCO_stop(t, y) : 
    return y[0] - (1/6) # stops if x == 1/6 (ISCO Schwarzschild)
ISCO_stop.terminal = True # stops if ISCO_stop == 0

# Solve the system using solve_ivp
sol = solve_ivp(taylort4_rhs, t_span=[T_0, T_F], t_eval=np.arange(T_0, T_F+DT, DT), y0=X_0, args=ARGS)