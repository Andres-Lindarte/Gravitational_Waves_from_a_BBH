"""
rol2_paraview.py -- Exportador a ParaView (Python PURO + pyevtk; nada de esto corre dentro de Numba).

    pip install pyevtk

Archivos generados en `outdir`
------------------------------
  bodies.pvd      + bodies/b_####.vtu     2 puntos por frame (m1, m2): velocity [c], mass_Msun,
                                          horizon_radius_km (=2Gm/c^2), f_gw_Hz, body_id
  trail_m1.vtu, trail_m2.vtu              trayectoria completa (polilínea): time_s, f_gw_Hz, x_pn
  wavefield.pvd   + wavefield/w_####.vti  ImageData 2D en el plano orbital: rh_plus_km (= R*h+ en km)
                                          y h_plus (adim.). ILUSTRATIVO: solo modo (2,2), campo lejano
                                          con retardo t - R/c, enmascarado dentro de la zona cercana.
  h_of_t.csv                              t_s, h_plus, h_cross, f_gw_Hz, x_pn
  paraview_manifest.json                  unidades y parámetros

NOTA sobre formatos: la API de alto nivel de pyevtk escribe puntos/líneas como .vtu (no .vtp) e imágenes
como .vti; .pvd se escribe con VtkGroup. ParaView abre todos igual.

Unidades: POSICIONES EN km (marco del detector, G m/c^2 = m_tot * 1.4766 km), tiempo del .pvd en SEGUNDOS
(t = 0 al final de la integración, así que los tiempos son negativos), velocidades en unidades de c.

Submuestreo: los frames son equiespaciados en FASE ORBITAL, no en tiempo (la chirp se acelera). En ParaView
usar "Snap To TimeSteps" en la barra de animación. El h_sim.npz para el Rol 3 NO se submuestrea.

Receta en ParaView
  1. File > Open bodies.pvd (Apply). Filters > Glyph: Sphere, Scale Array = horizon_radius_km, factor 1.
  2. Open trail_m1.vtu / trail_m2.vtu: Filters > Tube, colorear por f_gw_Hz.
  3. Open wavefield.pvd: colorear por rh_plus_km (la 1/R ya está quitada); opcional Warp By Scalar.
  4. La órbita (~10^3 km) es ~10x menor que la longitud de onda (~10^4 km): usa dos vistas enlazadas
     (View > Split) con zoom distinto, o un Glyph más grande.
"""
import json
import os
from math import pi
import numpy as np

import pn_coeffs as pnc

C_KM_S = 299792.458


def _pyevtk():
    try:
        from pyevtk.hl import pointsToVTK, polyLinesToVTK, imageToVTK
        from pyevtk.vtk import VtkGroup
    except ImportError as exc:
        raise ImportError("Falta pyevtk:  pip install pyevtk") from exc
    return pointsToVTK, polyLinesToVTK, imageToVTK, VtkGroup


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def phase_uniform_times(t_s, phi, n):
    """n instantes con la fase orbital equiespaciada (phi debe ser creciente)."""
    levels = np.linspace(phi[0], phi[-1], n)
    return np.interp(levels, phi, t_s)


def body_state(t_f, out, nu, m1, m2, km_per_code):
    """Posiciones [km], velocidades [c] de m1 y m2 en los instantes t_f (CM; r = m/γ(x))."""
    t, x, phi, xd = out["t"], out["x"], out["phi"], out["xdot"]
    xf = np.interp(t_f, t, x)
    pf = np.interp(t_f, t, phi)
    xdf = np.interp(t_f, t, xd)
    gam = pnc.gamma_of_x(xf, nu)
    r = 1.0 / gam                                    # unidades de código
    rdot = -pnc.dgamma_dx(xf, nu) / gam**2 * xdf
    om = xf**1.5
    n = np.stack([np.cos(pf), np.sin(pf), np.zeros_like(pf)], 1)
    lam = np.stack([-np.sin(pf), np.cos(pf), np.zeros_like(pf)], 1)
    dr = r[:, None] * n * km_per_code                # separación relativa en km
    dv = rdot[:, None] * n + (r * om)[:, None] * lam # v/c
    m = m1 + m2
    X1, X2 = m1 / m, m2 / m
    return dict(r1=X2 * dr, r2=-X1 * dr, v1=X2 * dv, v2=-X1 * dv, x=xf, phi=pf)


def wavefield_frame(t_f, X, Y, out, nu, km_per_code, near_km):
    """Campo (2,2) en el plano orbital: h+ = (2 G mu x/(c^2 R)) cos[2(ψ(t_ret) - Φ)],  t_ret = t - R/c.
    Devuelve (R*h+ [km], h+ adim.) con 0 fuera de la señal y dentro de la zona cercana."""
    R = np.hypot(X, Y)
    Phi = np.arctan2(Y, X)
    t_ret = t_f - R / C_KM_S
    ok = (t_ret >= out["t"][0]) & (R >= near_km)
    xr = np.interp(t_ret, out["t"], out["x"])
    psir = np.interp(t_ret, out["t"], out["psi"])
    G_mu_km = nu * km_per_code                       # G mu / c^2 en km
    rh = np.where(ok, 2.0 * G_mu_km * xr * np.cos(2.0 * (psir - Phi)), 0.0)
    h = np.where(ok, rh / np.maximum(R, 1e-9), 0.0)
    return rh, h


def export_paraview(out, outdir, n_frames=600, n_wave_frames=150, wave_n=161,
                    wave_extent_km=None, trail_points=3000):
    """Exporta resultados de rol2_engine.compute_rol2 a formatos de ParaView.

    out : dict devuelto por compute_rol2. Usa: t (s), x, xdot (código), phi, psi, f_gw (Hz),
          h_plus, h_cross y meta (m1_Msun, m2_Msun, nu, km_per_code).
    Devuelve dict con las rutas escritas."""
    pointsToVTK, polyLinesToVTK, imageToVTK, VtkGroup = _pyevtk()
    outdir = os.path.abspath(outdir)
    for sub in ("bodies", "wavefield"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    meta = out["meta"]
    nu, m1, m2 = meta["nu"], meta["m1_Msun"], meta["m2_Msun"]
    km = meta["km_per_code"]
    rs1, rs2 = 2 * m1 * km / (m1 + m2), 2 * m2 * km / (m1 + m2)   # 2 G m_i/c^2 en km
    written = {}

    # ---- cuerpos (serie temporal .pvd) ---------------------------------------
    t_frames = phase_uniform_times(out["t"], out["phi"], n_frames)
    st = body_state(t_frames, out, nu, m1, m2, km)
    f_fr = np.interp(t_frames, out["t"], out["f_gw"])
    grp = VtkGroup(os.path.join(outdir, "bodies"))
    for k, tf in enumerate(t_frames):
        pos = np.array([st["r1"][k], st["r2"][k]])
        vel = np.array([st["v1"][k], st["v2"][k]])
        path = pointsToVTK(
            os.path.join(outdir, "bodies", f"b_{k:04d}"),
            _f32(pos[:, 0]), _f32(pos[:, 1]), _f32(pos[:, 2]),
            data={"velocity": (_f32(vel[:, 0]), _f32(vel[:, 1]), _f32(vel[:, 2])),
                  "mass_Msun": _f32([m1, m2]),
                  "horizon_radius_km": _f32([rs1, rs2]),
                  "f_gw_Hz": _f32([f_fr[k], f_fr[k]]),
                  "body_id": _f32([0, 1])})
        grp.addFile(filepath=path, sim_time=float(tf))
    grp.save()
    written["bodies_pvd"] = os.path.join(outdir, "bodies.pvd")

    # ---- trayectorias completas ----------------------------------------------
    idx = np.unique(np.linspace(0, len(out["t"]) - 1, min(trail_points, len(out["t"]))).astype(int))
    tt = out["t"][idx]
    s_all = body_state(tt, out, nu, m1, m2, km)
    for name, key in (("trail_m1", "r1"), ("trail_m2", "r2")):
        p = s_all[key]
        written[name] = polyLinesToVTK(
            os.path.join(outdir, name),
            _f32(p[:, 0]), _f32(p[:, 1]), _f32(p[:, 2]),
            pointsPerLine=np.array([len(idx)], dtype=np.int32),
            pointData={"time_s": _f32(tt), "f_gw_Hz": _f32(out["f_gw"][idx]), "x_pn": _f32(out["x"][idx])})

    # ---- campo de ondas en el plano orbital (.vti + .pvd) --------------------
    f0 = float(out["f_gw"][0])
    lam0 = C_KM_S / f0
    near_km = lam0 / (2 * pi)
    L = wave_extent_km if wave_extent_km is not None else 2.5 * lam0
    xs = np.linspace(-L, L, wave_n)
    dxg = xs[1] - xs[0]
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    tw = phase_uniform_times(out["t"], out["phi"], n_wave_frames)
    gw = VtkGroup(os.path.join(outdir, "wavefield"))
    for k, tf in enumerate(tw):
        rh, h = wavefield_frame(tf, XX, YY, out, nu, km, near_km)
        path = imageToVTK(
            os.path.join(outdir, "wavefield", f"w_{k:04d}"),
            origin=(-L, -L, 0.0), spacing=(dxg, dxg, 1.0),
            pointData={"rh_plus_km": np.asfortranarray(rh[:, :, None], dtype=np.float32),
                       "h_plus": np.asfortranarray(h[:, :, None], dtype=np.float32)})
        gw.addFile(filepath=path, sim_time=float(tf))
    gw.save()
    written["wavefield_pvd"] = os.path.join(outdir, "wavefield.pvd")

    # ---- tabla y manifiesto ----------------------------------------------------
    csv_path = os.path.join(outdir, "h_of_t.csv")
    np.savetxt(csv_path,
               np.column_stack([out["t"], out["h_plus"], out["h_cross"], out["f_gw"], out["x"]]),
               delimiter=",", header="t_s,h_plus,h_cross,f_gw_Hz,x_pn", comments="")
    written["csv"] = csv_path
    manifest = dict(units=dict(position="km", time="s (t=0 al final de la integracion)", velocity="c",
                               wavefield="plano orbital, modo (2,2), campo lejano ilustrativo"),
                    n_body_frames=int(n_frames), n_wave_frames=int(n_wave_frames),
                    near_zone_mask_km=near_km, wave_extent_km=L, meta={k: v for k, v in meta.items()})
    with open(os.path.join(outdir, "paraview_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    return written
