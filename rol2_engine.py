"""
rol2_engine.py -- Motor de cálculo del Rol 2: h+(t), hx(t) y balance energético hasta 3.5PN.

Kernels @njit(cache=True):  energy_x, denergy_dx, flux_x, waveform_kernel, cumint4, energy_flux_arrays.
Todo el cálculo usa G = c = 1 y m = m1+m2 = 1; el escalado físico lo hace rol2_io.to_physical().
Si Numba no está instalado, el mismo código corre en Python puro (lento pero idéntico).

Interfaz principal:  compute_rol2(rol1: dict, cfg: Rol2Config) -> dict
"""
from dataclasses import dataclass, field
from math import pi
import numpy as np
from scipy.interpolate import CubicSpline

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                       # modo de reserva: Python puro
    HAVE_NUMBA = False
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f
    prange = range

import pn_coeffs as pnc
import rol2_io as rio


# =============================================================================
# Kernels Numba
# =============================================================================
@njit(cache=True)
def energy_x(x, e, nu):
    """E(x)/(m c^2) = -(nu x/2) sum_n e[n] x^n   (e: ver pn_coeffs.energy_coeffs)."""
    s = 0.0
    for n in range(e.shape[0] - 1, -1, -1):
        s = s * x + e[n]
    return -0.5 * nu * x * s


@njit(cache=True)
def denergy_dx(x, e, nu):
    s = 0.0
    for n in range(e.shape[0] - 1, -1, -1):
        s = s * x + (n + 1) * e[n]
    return -0.5 * nu * s


@njit(cache=True)
def flux_x(x, f, fl, nu):
    """F(x) en unidades de c^5/G:  (32/5) nu^2 x^5 sum_k (f[k] + fl[k] ln(16x)) sqrt(x)^k."""
    y = np.sqrt(x)
    L = np.log(16.0 * x)
    s = 0.0
    for k in range(f.shape[0] - 1, -1, -1):
        s = s * y + f[k] + L * fl[k]
    return 6.4 * nu * nu * x**5 * s


@njit(cache=True, parallel=True)
def waveform_kernel(x, xdot, psi, psidot, C, CL, mm, Yp, Ym, nu, hp, hc, flux_h):
    """Suma de modos h_lm y flujo irradiado desde los modos.

    Entradas (todas float64 salvo C, CL, Yp, Ym complex128 y mm int64):
        x, xdot, psi, psidot : (N,)   x(t), dx/dt, fase ψ(t) y dψ/dt   [unidades de código]
        C, CL                : (NM,8) coeficientes de y^k y de y^k·ln(16x) de H_lm (con prefactor)
        mm                   : (NM,)  m de cada modo (m>0)
        Yp, Ym               : (NM,)  factores angulares (pn_coeffs.angular_factors)
    Salidas (se llenan in situ; R = 1, m = 1):
        hp, hc  : (N,)  h+ y hx  (multiplicar por G m/(c^2 D_L) para pasar a deformación)
        flux_h  : (N,)  dE/dt = (R^2/16π) sum_{lm} |dh_lm/dt|^2  (ambos signos de m)
    h_lm = 2ν x sqrt(16π/5) H_lm(x) e^{-i m ψ};  dh_lm/dt = (A' ẋ - i m ψ̇ A) e^{-i m ψ}.
    """
    n = x.shape[0]
    NM = C.shape[0]
    NK = C.shape[1]
    pref = 2.0 * nu * np.sqrt(16.0 * np.pi / 5.0)
    for i in prange(n):
        xi = x[i]
        y = np.sqrt(xi)
        L = np.log(16.0 * xi)
        X = 0.0 + 0.0j
        fsum = 0.0
        for j in range(NM):
            H = 0.0 + 0.0j
            dH = 0.0 + 0.0j
            for k in range(NK):
                c0 = C[j, k]
                c1 = CL[j, k]
                if c0 == 0.0 and c1 == 0.0:
                    continue
                ck = c0 + L * c1
                H += ck * y**k
                dH += (0.5 * k * ck + c1) * y**(k - 2)
            A = pref * xi * H
            Ap = pref * (H + xi * dH)
            ph = np.exp(-1j * mm[j] * psi[i])
            hlm = A * ph
            X += hlm * Yp[j] + np.conj(hlm) * Ym[j]
            hd = (Ap * xdot[i] - 1j * mm[j] * psidot[i] * A) * ph
            fsum += hd.real * hd.real + hd.imag * hd.imag
        hp[i] = X.real
        hc[i] = -X.imag
        flux_h[i] = 2.0 * fsum / (16.0 * np.pi)


@njit(cache=True)
def cumint4(y, dt):
    """Integral acumulada de 4.º orden (cúbica de Lagrange por intervalo, rejilla uniforme), 0 en la 1.ª muestra.
    Error global O(dt^4) (el trapecio, O(dt^2), daba ~1e-5 en ε_F a 4096 Hz)."""
    n = y.shape[0]
    out = np.zeros(n)
    for i in range(n - 1):
        if n < 4:
            seg = 12.0 * (y[i] + y[i + 1])
        elif i == 0:
            seg = 9.0 * y[0] + 19.0 * y[1] - 5.0 * y[2] + y[3]
        elif i == n - 2:
            seg = y[n - 4] - 5.0 * y[n - 3] + 19.0 * y[n - 2] + 9.0 * y[n - 1]
        else:
            seg = -y[i - 1] + 13.0 * y[i] + 13.0 * y[i + 1] - y[i + 2]
        out[i + 1] = out[i] + dt * seg / 24.0
    return out


@njit(cache=True)
def energy_flux_arrays(x, e, f, fl, nu):
    n = x.shape[0]
    E = np.empty(n)
    F = np.empty(n)
    for i in range(n):
        E[i] = energy_x(x[i], e, nu)
        F[i] = flux_x(x[i], f, fl, nu)
    return E, F


# =============================================================================
# Configuración y utilidades (Python)
# =============================================================================
@dataclass
class Rol2Config:
    m1_Msun: float = None            # si None se toma de la entrada del Rol 1 (marco detector)
    m2_Msun: float = None
    DL_Mpc: float = 410.0            # distancia de luminosidad fiducial
    iota: float = 0.0                # inclinación [rad]
    phi_c: float = 0.0               # fase ψ en t = 0 [rad]
    phi_obs: float = pi / 2          # azimut del observador (convención BFIS08: π/2)
    fs: float = 4096.0               # Hz
    flux_pn: int = 7                 # semiórdenes del flujo   (7 = 3.5PN)
    energy_pn: int = 6               # semiórdenes de E(x)     (6 = 3PN; no hay 3.5PN conservativo)
    amp_pn: int = 6                  # semiórdenes de la amplitud (6 = 3PN; ver pn_coeffs)
    include_tail: bool = True        # fase ψ con el log de cola (False -> ψ = φ)
    x0_tail: float = None            # x0 del log; None -> x final (solo fija el origen temporal)
    taper_start_s: float = 0.10
    taper_end_s: float = 0.02


def tail_phase(x, xdot, phi, phidot, nu, x0):
    """ψ = φ - 3 x^{3/2} (1 - ν x/2) ln(x/x0)   (BFIS08 Eq. 8.8) y su derivada.
    Cambiar x0 equivale a un corrimiento temporal: δψ ∝ ω."""
    lx = np.log(x / x0)
    g = 3.0 * x**1.5 * (1.0 - 0.5 * nu * x) * lx
    gp = 3.0 * ((1.5 * np.sqrt(x) - 1.25 * nu * x**1.5) * lx + np.sqrt(x) * (1.0 - 0.5 * nu * x))
    return phi - g, phidot - gp * xdot


def resample_inspiral(t_code, x, phi, m_tot, fs):
    """Remuestrea (x, φ) del Rol 1 a una rejilla UNIFORME de fs Hz con t = 0 en la última muestra.
    Devuelve t_s, t_code_out, x, dx/dt, φ, dφ/dt (derivadas por spline cúbico, en unidades de código)."""
    s_per_code = m_tot * rio.T_SUN
    T_s = (t_code[-1] - t_code[0]) * s_per_code
    n = int(np.floor(T_s * fs + 1e-9)) + 1
    t_s = (np.arange(n) - (n - 1)) / fs                    # exactamente 1/fs, termina en 0.0
    tc = t_code[-1] + t_s / s_per_code
    sx, sp = CubicSpline(t_code, x), CubicSpline(t_code, phi)
    return t_s, tc, sx(tc), sx(tc, 1), sp(tc), sp(tc, 1)


def apply_taper(h, fs, t_start_s, t_end_s):
    """Ventana de Tukey asimétrica (coseno elevado) en los extremos."""
    w = np.ones_like(h)
    n0, n1 = int(round(t_start_s * fs)), int(round(t_end_s * fs))
    if n0 > 1:
        w[:n0] = 0.5 * (1 - np.cos(np.pi * np.arange(n0) / n0))
    if n1 > 1:
        w[-n1:] = 0.5 * (1 + np.cos(np.pi * np.arange(1, n1 + 1) / n1))
    return h * w


# =============================================================================
# Interfaz principal
# =============================================================================
def compute_rol2(rol1, cfg=None):
    """Calcula h+(t), hx(t) y el balance energético a partir de la salida del Rol 1.

    rol1 : dict (ver rol2_io.load_rol1)
        t     (N,)  tiempo en G m/c^3     [creciente]
        x_pn  (N,)  parámetro PN x         [adim.]
        phi   (N,)  fase orbital acumulada [rad]
        m1_Msun, m2_Msun                   [marco del detector]
    Devuelve dict con:
        t (s), h_plus, h_cross (adim., ya escalados a D_L y con taper), f_gw (Hz), x, phi, psi,
        E_orb_mc2, E_rad_F_mc2, E_rad_h_mc2, eps_F, eps_h, *_Msun (M☉ c^2), M_sys_Msun, flux_F, flux_h,
        meta (dict con todos los parámetros)
    """
    cfg = cfg or Rol2Config()
    m1 = cfg.m1_Msun if cfg.m1_Msun is not None else float(rol1["m1_Msun"])
    m2 = cfg.m2_Msun if cfg.m2_Msun is not None else float(rol1["m2_Msun"])
    m, nu, delta = rio.mass_params(m1, m2)
    phys = rio.to_physical(m, cfg.DL_Mpc)

    t_s, tc, x, xd, phi, phid = resample_inspiral(
        np.asarray(rol1["t"], float), np.asarray(rol1["x_pn"], float),
        np.asarray(rol1["phi"], float), m, cfg.fs)
    dtc = tc[1] - tc[0]

    # --- tablas de coeficientes (Python) -> arreglos para los kernels -------
    e = pnc.energy_coeffs(nu, cfg.energy_pn)
    f, fl = pnc.flux_coeffs(nu, cfg.flux_pn)
    ell, mm, C, CL = pnc.mode_table(nu, delta, cfg.amp_pn)
    Yp, Ym = pnc.angular_factors(ell, mm, cfg.iota, cfg.phi_obs)

    # --- fase ψ ---------------------------------------------------------------
    if cfg.include_tail:
        x0 = cfg.x0_tail if cfg.x0_tail is not None else x[-1]
        psi, psid = tail_phase(x, xd, phi, phid, nu, x0)
    else:
        psi, psid = phi.copy(), phid.copy()
    psi = psi + (cfg.phi_c - psi[-1])

    # --- suma de modos y flujo ------------------------------------------------
    hp = np.empty_like(x); hc = np.empty_like(x); flux_h = np.empty_like(x)
    waveform_kernel(np.ascontiguousarray(x), np.ascontiguousarray(xd),
                    np.ascontiguousarray(psi), np.ascontiguousarray(psid),
                    C, CL, mm, Yp, Ym, nu, hp, hc, flux_h)

    # --- balance energético ---------------------------------------------------
    E, F = energy_flux_arrays(np.ascontiguousarray(x), e, f, fl, nu)
    E_rad_F = cumint4(F, dtc)
    E_rad_h = cumint4(flux_h, dtc)
    dE = E[0] - E                                         # energía perdida por la órbita (>0)
    with np.errstate(invalid="ignore", divide="ignore"):
        eps_F = np.where(dE > 0, 1.0 - E_rad_F / dE, np.nan)
        eps_h = np.where(dE > 0, 1.0 - E_rad_h / dE, np.nan)

    # --- unidades físicas -----------------------------------------------------
    s = phys["strain_scale"]
    h_plus = apply_taper(hp * s, cfg.fs, cfg.taper_start_s, cfg.taper_end_s)
    h_cross = apply_taper(hc * s, cfg.fs, cfg.taper_start_s, cfg.taper_end_s)
    f_gw = phid / (pi * phys["s_per_code"])               # f_gw = omega/pi  [Hz]

    meta = dict(m1_Msun=m1, m2_Msun=m2, m_tot_Msun=m, nu=nu, delta=delta,
                DL_Mpc=cfg.DL_Mpc, iota=cfg.iota, phi_c=cfg.phi_c, phi_obs=cfg.phi_obs,
                fs=cfg.fs, pn_order_flux=cfg.flux_pn, pn_order_energy=cfg.energy_pn,
                pn_order_amp=cfg.amp_pn, include_tail=int(cfg.include_tail),
                x_start=float(x[0]), x_max=float(x[-1]),
                f_gw_start=float(f_gw[0]), f_gw_end=float(f_gw[-1]),
                taper_start_s=cfg.taper_start_s, taper_end_s=cfg.taper_end_s,
                strain_scale=s, s_per_code=phys["s_per_code"], km_per_code=phys["km_per_code"],
                t_zero="final de la integracion (~t_c); t en segundos, dt=1/fs exacto")
    return dict(
        t=t_s, h_plus=h_plus, h_cross=h_cross, f_gw=f_gw, x=x, phi=phi, psi=psi,
        t_code=tc, xdot=xd, flux_F=F, flux_h=flux_h,
        E_orb_mc2=E, E_rad_F_mc2=E_rad_F, E_rad_h_mc2=E_rad_h, eps_F=eps_F, eps_h=eps_h,
        E_orb_Msun=E * m, E_rad_F_Msun=E_rad_F * m, E_rad_h_Msun=E_rad_h * m,
        M_sys_Msun=m * (1.0 + E),
        meta=meta)


# =============================================================================
# Cuadrupolo general (0PN) a partir de la cinemática del Rol 1  -- línea base del PDF
# =============================================================================
def h_quadrupole_from_kinematics(r_rel, v_rel, a_rel, mu, iota, phi_obs=pi / 2):
    """h+ y hx de la fórmula de Einstein para una trayectoria planar GENERAL (sin suponer órbita circular).

    r_rel, v_rel, a_rel : (N,3)  posición, velocidad y aceleración RELATIVAS r1-r2 (G=c=1)
    mu                  : masa reducida (en unidades de m)
    Devuelve (h+, hx) con R = 1.  I''_ij = 2 mu (v_i v_j + r_(i a_j)).
    Triada (BFIS08): N = (sinι cosΦ, sinι sinΦ, cosι), P = N×Z/|N×Z|, Q = N×P."""
    N = np.array([np.sin(iota) * np.cos(phi_obs), np.sin(iota) * np.sin(phi_obs), np.cos(iota)])
    z = np.array([0.0, 0.0, 1.0])
    P = np.cross(N, z)
    P = P / np.linalg.norm(P) if np.linalg.norm(P) > 1e-12 else np.array([np.sin(phi_obs), -np.cos(phi_obs), 0.0])
    Q = np.cross(N, P)
    Iij = 2.0 * mu * (np.einsum("ni,nj->nij", v_rel, v_rel)
                      + 0.5 * (np.einsum("ni,nj->nij", r_rel, a_rel) + np.einsum("ni,nj->nij", a_rel, r_rel)))
    hp = np.einsum("nij,i,j->n", Iij, P, P) - np.einsum("nij,i,j->n", Iij, Q, Q)
    hc = np.einsum("nij,i,j->n", Iij, P, Q) + np.einsum("nij,i,j->n", Iij, Q, P)
    return hp, hc
