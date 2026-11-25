"""
mock_rol1.py -- SUSTITUTO del Rol 1 (NO es su código). Sirve para (a) probar el Rol 2 de extremo a extremo y
(b) mostrar el contrato de datos con un ejemplo ejecutable.

Opción B acordada: ODE de balance de flujo en (x, φ), integrada con RK4:
    dx/dt = -F(x)/E'(x)          (rr_scheme='T1', TaylorT1)
    dx/dt = (64/5) ν x^5 Σ ...    (rr_scheme='T4', TaylorT4 truncado)   |  '0PN' (solo cuadrupolo)
    dφ/dt = x^(3/2)                (G = c = m = 1)
r1, r2, v1, v2 se reconstruyen con r = m/γ(x) (solo para visualización).
"""
import numpy as np
from math import pi
import pn_coeffs as pnc
import rol2_io as rio
from rol2_engine import njit, denergy_dx, flux_x

SCHEMES = {"0PN": 0, "T1": 1, "T4": 2}


@njit(cache=True)
def _rhs_x(x, scheme, e, f, fl, t4, t4l, nu):
    if scheme == 0:
        return 12.8 * nu * x**5
    if scheme == 1:
        return -flux_x(x, f, fl, nu) / denergy_dx(x, e, nu)
    y = np.sqrt(x)
    L = np.log(16.0 * x)
    s = 0.0
    for k in range(t4.shape[0] - 1, -1, -1):
        s = s * y + t4[k] + L * t4l[k]
    return 12.8 * nu * x**5 * s


@njit(cache=True)
def rk4_inspiral(x0, x_max, h, scheme, e, f, fl, t4, t4l, nu, nmax):
    """RK4 de paso fijo h para (x, φ). Se detiene antes de superar x_max."""
    t = np.empty(nmax); x = np.empty(nmax); phi = np.empty(nmax)
    t[0] = 0.0; x[0] = x0; phi[0] = 0.0
    n = 1
    while n < nmax:
        xn = x[n - 1]
        k1 = _rhs_x(xn, scheme, e, f, fl, t4, t4l, nu);                p1 = xn**1.5
        x2 = xn + 0.5 * h * k1
        k2 = _rhs_x(x2, scheme, e, f, fl, t4, t4l, nu);                p2 = x2**1.5
        x3 = xn + 0.5 * h * k2
        k3 = _rhs_x(x3, scheme, e, f, fl, t4, t4l, nu);                p3 = x3**1.5
        x4 = xn + h * k3
        k4 = _rhs_x(x4, scheme, e, f, fl, t4, t4l, nu);                p4 = x4**1.5
        xnew = xn + h / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        if xnew >= x_max:
            break
        x[n] = xnew
        phi[n] = phi[n - 1] + h / 6.0 * (p1 + 2*p2 + 2*p3 + p4)
        t[n] = t[n - 1] + h
        n += 1
    return t[:n], x[:n], phi[:n]


def run_mock_rol1(m1_Msun=39.2, m2_Msun=31.6, f_gw0=20.0, x_max=1/6, dt=0.25,
                  rr_scheme="T1", pn_order=7, nmax=400000):
    m, nu, delta = rio.mass_params(m1_Msun, m2_Msun)
    s_per_code = m * rio.T_SUN
    x0 = (pi * f_gw0 * s_per_code) ** (2.0 / 3.0)              # x = (m ω)^(2/3), ω = π f_gw
    e = pnc.energy_coeffs(nu, pn_order)
    f, fl = pnc.flux_coeffs(nu, pn_order)
    t4, t4l = pnc.dxdt_t4_coeffs(nu, pn_order)
    t, x, phi = rk4_inspiral(x0, x_max, dt, SCHEMES[rr_scheme], e, f, fl, t4, t4l, nu, nmax)
    # reconstrucción de la cinemática (solo visualización)
    xd = np.array([_rhs_x(xi, SCHEMES[rr_scheme], e, f, fl, t4, t4l, nu) for xi in x])
    gam = pnc.gamma_of_x(x, nu)
    r = 1.0 / gam
    rdot = -pnc.dgamma_dx(x, nu) / gam**2 * xd
    om = x**1.5
    nvec = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], 1)
    lam = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], 1)
    dr = r[:, None] * nvec
    dv = rdot[:, None] * nvec + (r * om)[:, None] * lam
    X1, X2 = m1_Msun / m, m2_Msun / m
    return dict(t=t, x_pn=x, phi=phi, r1=X2 * dr, r2=-X1 * dr, v1=X2 * dv, v2=-X1 * dv,
                m1_Msun=m1_Msun, m2_Msun=m2_Msun, rr_scheme=rr_scheme, integrator="RK4", dt=dt,
                f_gw0=f_gw0, pn_order_dyn=pn_order)


def save_rol1(path, d):
    np.savez(path, **{k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()})


if __name__ == "__main__":
    d = run_mock_rol1()
    save_rol1("rol1_output.npz", d)
    print(f"rol1_output.npz: N={len(d['t'])}, x: {d['x_pn'][0]:.4f} -> {d['x_pn'][-1]:.4f}")
