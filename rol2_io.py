"""
rol2_io.py -- Unidades, entrada del Rol 1 (validación) y salidas del Rol 2.

Entrada del Rol 1  (rol1_output.npz, arreglos float64 C-contiguos)
------------------------------------------------------------------
    t        (N,)    G m/c^3   creciente (puede ser no uniforme)
    x_pn     (N,)    adim.     x = (G m omega/c^3)^(2/3)          [Opción B]
    phi      (N,)    rad       fase orbital ACUMULADA (sin envolver)
    r1, r2   (N,3)   G m/c^2   posiciones en el sistema CM        [solo visualización]
    v1, v2   (N,3)   c         velocidades                        [solo visualización]
    m1_Msun, m2_Msun  escalares (marco del detector); opcionales: rr_scheme, integrator, dt, f_gw0
  Si faltan x_pn/phi se derivan de r1,r2,v1,v2 (órbita en z=0):
    phi = unwrap(atan2(dy,dx)),  omega = (dx*dvy - dy*dvx)/r^2,  x = omega^(2/3)
"""
import json
from math import pi
import numpy as np

# ---- constantes (SI) --------------------------------------------------------
C_SI     = 299792458.0
G_SI     = 6.67430e-11
GM_SUN   = 1.32712440018e20            # m^3 s^-2
MSUN_KG  = GM_SUN / G_SI
MPC_M    = 3.0856775814913673e22
T_SUN    = GM_SUN / C_SI**3            # G Msun/c^3  [s]      ~ 4.9255e-6
R_SUN_M  = GM_SUN / C_SI**2            # G Msun/c^2  [m]      ~ 1476.6
R_SUN_KM = R_SUN_M / 1e3
E_SUN_J  = MSUN_KG * C_SI**2           # Msun c^2    [J]


# ---- escalado a unidades físicas: UNA sola función para todo el equipo ------
def mass_params(m1, m2):
    m = m1 + m2
    nu = m1 * m2 / m**2
    delta = (m1 - m2) / m
    return m, nu, delta


def to_physical(m_tot_Msun, DL_Mpc=None):
    """Factores para pasar de unidades G=c=m=1 a SI. Rol 3 y ParaView usan estos mismos números.
        t_phys[s]  = t_code  * s_per_code        (s_per_code = G m/c^3)
        L_phys[km] = L_code  * km_per_code       (km_per_code = G m/c^2)
        h          = h_code  * strain_scale      (strain_scale = G m/(c^2 D_L))
        E[Msun c^2]= E_code  * m_tot_Msun"""
    out = dict(s_per_code=m_tot_Msun * T_SUN,
               km_per_code=m_tot_Msun * R_SUN_KM,
               Msun_per_code=m_tot_Msun)
    if DL_Mpc is not None:
        out["strain_scale"] = m_tot_Msun * R_SUN_M / (DL_Mpc * MPC_M)
    return out


# ---- entrada del Rol 1 -------------------------------------------------------
def load_rol1(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ("m1_Msun", "m2_Msun", "dt", "f_gw0"):
        if k in d:
            d[k] = float(d[k])
    return normalize_rol1(d)


def normalize_rol1(d):
    d = dict(d)
    if "x_pn" not in d or "phi" not in d:
        need = ("r1", "r2", "v1", "v2")
        if not all(k in d for k in need):
            raise ValueError("Rol 1 debe entregar (x_pn, phi) o bien (r1, r2, v1, v2).")
        dr = np.asarray(d["r1"]) - np.asarray(d["r2"])
        dv = np.asarray(d["v1"]) - np.asarray(d["v2"])
        r2 = dr[:, 0]**2 + dr[:, 1]**2
        omega = (dr[:, 0]*dv[:, 1] - dr[:, 1]*dv[:, 0]) / r2
        d["phi"] = np.unwrap(np.arctan2(dr[:, 1], dr[:, 0]))
        d["x_pn"] = np.abs(omega)**(2.0/3.0)
    return d


def validate_rol1(d, min_steps_per_orbit=50):
    """Lanza ValueError con un mensaje claro si la entrada incumple el contrato."""
    t, x, phi = (np.asarray(d[k], dtype=float) for k in ("t", "x_pn", "phi"))
    errs = []
    if not (t.ndim == x.ndim == phi.ndim == 1 and len(t) == len(x) == len(phi)):
        errs.append("t, x_pn, phi deben ser 1-D de igual longitud")
    if len(t) < 10:
        errs.append("muy pocos puntos")
    for name, a in (("t", t), ("x_pn", x), ("phi", phi)):
        if not np.all(np.isfinite(a)):
            errs.append(f"{name} contiene NaN/inf")
    if not errs:
        if not np.all(np.diff(t) > 0):
            errs.append("t no es estrictamente creciente")
        if not np.all(np.diff(phi) > 0):
            errs.append("phi no es monótona creciente (¿la envolvieron con mod 2*pi?)")
        if np.any(x <= 0) or np.any(x > 0.5):
            errs.append("x_pn fuera de (0, 0.5]: ¿está en G=c=m=1? (x = (m*omega)^(2/3))")
        steps = 2*pi / np.median(np.diff(phi))
        if steps < min_steps_per_orbit:
            errs.append(f"solo ~{steps:.0f} pasos por órbita (< {min_steps_per_orbit})")
        if not np.all(np.diff(x) > 0):
            errs.append("x_pn no crece con t: ¿incluyeron la reacción de radiación?")
    for k in ("m1_Msun", "m2_Msun"):
        if k not in d:
            errs.append(f"falta {k}")
    if errs:
        raise ValueError("Entrada del Rol 1 inválida:\n  - " + "\n  - ".join(errs))
    return True


# ---- salidas del Rol 2 -------------------------------------------------------
def save_h_sim(path, out, meta):
    """h_sim.npz  ->  Rol 3.  Arreglos float64; `meta_json` con todos los parámetros."""
    np.savez(path,
             t=out["t"], h_plus=out["h_plus"], h_cross=out["h_cross"], f_gw=out["f_gw"],
             **{k: np.float64(v) for k, v in meta.items()
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)},
             meta_json=json.dumps(meta, default=float))


def save_energy_balance(path, out):
    keys = ("t", "x", "E_orb_mc2", "E_rad_F_mc2", "E_rad_h_mc2", "eps_F", "eps_h",
            "E_orb_Msun", "E_rad_F_Msun", "E_rad_h_Msun", "M_sys_Msun", "flux_F", "flux_h")
    np.savez(path, **{k: out[k] for k in keys})
