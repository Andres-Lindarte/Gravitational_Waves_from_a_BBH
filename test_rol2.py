"""
test_rol2.py -- Validación del Rol 2.   Ejecutar:  python test_rol2.py   (o:  pytest -q test_rol2.py)

  A. Coeficientes      : ẋ de TaylorT4 (derivado de mis E y F) vs. literatura; flujo de los modos vs F(x) hasta 3PN.
  B. Armónicos         : _{-2}Y_lm contra formas cerradas y normalización.
  C. Caso base 0PN     : integrador RK4 vs. solución analítica de Peters; h± vs. fórmula del cuadrupolo
                         aplicada a la CINEMÁTICA (r, v, a), que verifica signos/convención de h×.
  D. Balance energético: ε_F (integración) y ε_h (independiente, desde los modos).
  E. Contrato de salida: rejilla 4096 Hz exacta, t=0 al final, sin NaN, taper.
  F. PyCBC TaylorT4    : se omite si PyCBC no está instalado (umbrales PROVISIONALES: calibrar en la 1.ª corrida).
  G. Exportador        : prueba de humo, se omite si pyevtk no está instalado.
"""
import os, sys, tempfile, traceback
from math import pi, sqrt
from unittest import SkipTest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pn_coeffs as pnc
import rol2_io as rio
import rol2_engine as eng
import mock_rol1 as mk

G_E = pnc.EULER_GAMMA
NUS = (0.08, 0.2, 0.25)


# ------------------------------------------------------------------ A
def _t4_literature(nu):
    t = np.zeros(8)
    t[0] = 1; t[2] = -743/336 - 11*nu/4; t[3] = 4*pi
    t[4] = 34103/18144 + 13661*nu/2016 + 59*nu**2/18
    t[5] = -(4159/672 + 189*nu/8)*pi
    t[6] = (16447322263/139708800 - 1712*G_E/105 + 16*pi**2/3 + (-56198689/217728 + 451*pi**2/48)*nu
            + 541*nu**2/896 - 5605*nu**3/2592)
    t[7] = (-4415/4032 + 358675*nu/6048 + 91495*nu**2/1512)*pi
    return t


def test_A1_taylorT4_series_matches_literature():
    for nu in NUS:
        t, tl = pnc.dxdt_t4_coeffs(nu)
        assert np.max(np.abs(t - _t4_literature(nu))) < 1e-10, nu
        assert abs(tl[6] + 856/105) < 1e-14


def test_A2_mode_flux_reproduces_F_through_3PN():
    """sum_{l,m>0} (m/2)^2 |H_lm|^2 = F/F_N  coeficiente a coeficiente hasta y^6 (todos los ν, con log)."""
    L = 0.37
    for nu in NUS:
        ell, mm, C, CL = pnc.mode_table(nu, sqrt(1 - 4*nu), 6)
        tot = np.zeros(8)
        for j in range(len(ell)):
            H = C[j] + L*CL[j]
            tot += (mm[j]/2)**2 * np.convolve(H, np.conj(H))[:8].real
        f, fl = pnc.flux_coeffs(nu)
        assert np.max(np.abs(tot[:7] - (f + L*fl)[:7])) < 1e-9, (nu, tot[:7] - (f + L*fl)[:7])


# ------------------------------------------------------------------ B
def test_B1_spin_weighted_harmonics():
    th, ph = 0.9, 0.4
    c = np.cos(th)
    assert abs(pnc.swsh_m2(2, 2, th, ph) - sqrt(5/(64*pi))*(1 + c)**2*np.exp(2j*ph)) < 1e-14
    assert abs(pnc.swsh_m2(2, -2, th, ph) - sqrt(5/(64*pi))*(1 - c)**2*np.exp(-2j*ph)) < 1e-14
    xs, ws = np.polynomial.legendre.leggauss(60)             # ∫|Y|^2 dΩ = 1
    for (l, m) in pnc.MODES:
        val = sum(w*abs(pnc.swsh_m2(l, m, np.arccos(u), 0.0))**2 for u, w in zip(xs, ws))*2*pi
        assert abs(val - 1) < 1e-10, (l, m, val)


# ------------------------------------------------------------------ C
def test_C1_rk4_vs_peters_0PN():
    d = mk.run_mock_rol1(rr_scheme="0PN", dt=0.25)
    m, nu, _ = rio.mass_params(39.2, 31.6)
    tau = d["t"][-1] - d["t"] + (5/(256*nu))*d["x_pn"][-1]**-4/ 1.0   # τ = t_c - t con x(τ) = (5/(256 ν τ))^(1/4)
    x_an = (5/(256*nu*tau))**0.25
    assert np.max(np.abs(d["x_pn"]/x_an - 1)) < 1e-6
    phi_an = -(8/5)*(5/(256*nu))**(3/8)*tau**(5/8)
    dphi = (d["phi"] - phi_an); dphi -= dphi[-1]
    assert np.max(np.abs(dphi)) < 1e-5


def _kinematics_on_grid(out):
    t = out["t_code"]; dt = t[1] - t[0]
    ps = out["psi"]          # con include_tail=False, ψ = φ - φ_final (fase de coalescencia φ_c = 0)
    r = (1.0/out["x"])[:, None]*np.stack([np.cos(ps), np.sin(ps), 0*ps], 1)
    v = np.gradient(r, dt, axis=0, edge_order=2)
    a = np.gradient(v, dt, axis=0, edge_order=2)
    return r, v, a


def test_C2_h_0PN_matches_closed_form_and_kinematic_quadrupole():
    d = mk.run_mock_rol1(rr_scheme="0PN", dt=0.25)
    iota = 1.0
    cfg = eng.Rol2Config(iota=iota, amp_pn=0, include_tail=False, taper_start_s=0, taper_end_s=0)
    out = eng.compute_rol2(d, cfg)
    m, nu, _ = rio.mass_params(39.2, 31.6)
    s = out["meta"]["strain_scale"]
    # (i) forma cerrada del 0PN en la convención de BFIS08 (observador en Φ=π/2)
    hp_cf = -2*nu*out["x"]*(1 + np.cos(iota)**2)*np.cos(2*out["psi"])
    hc_cf = -4*nu*out["x"]*np.cos(iota)*np.sin(2*out["psi"])
    assert np.max(np.abs(out["h_plus"]/s - hp_cf)) < 1e-12
    assert np.max(np.abs(out["h_cross"]/s - hc_cf)) < 1e-12
    # (ii) fórmula de Einstein sobre la cinemática (r, v, a) -> comprueba el signo de h×
    r, v, a = _kinematics_on_grid(out)
    hp_q, hc_q = eng.h_quadrupole_from_kinematics(r, v, a, nu, iota, pi/2)
    sl = slice(5, -5)
    ref = np.sqrt(np.mean(hp_q[sl]**2 + hc_q[sl]**2))
    err = np.sqrt(np.mean((hp_q[sl] - out["h_plus"][sl]/s)**2 + (hc_q[sl] - out["h_cross"][sl]/s)**2))
    # la diferencia residual es el término de reacción de radiación (~2.5PN) que el 0PN de h no lleva
    print(f"   [C2] error relativo cuadrupolo-cinemático vs modo 22 (0PN): {err/ref:.4f}")
    assert err/ref < 0.03, err/ref


# ------------------------------------------------------------------ D
def test_D1_energy_balance():
    out = eng.compute_rol2(mk.run_mock_rol1(rr_scheme="T1", dt=0.5), eng.Rol2Config())
    assert np.nanmax(np.abs(out["eps_F"])) < 1e-7                         # solo error numérico (RK4+cuadratura)
    dM = out["M_sys_Msun"][0] - out["M_sys_Msun"]
    assert abs(dM[-1]/out["E_rad_F_Msun"][-1] - 1) < 1e-7                  # M(t0)-M(t) = E_rad/c^2
    x = out["x"]; eh = np.abs(out["eps_h"])
    assert np.nanmax(eh[x < 0.10]) < 0.015                                  # normalización de h verificada a <1.5%
    assert eh[-1] < 0.06                                                    # al ISCO domina el truncamiento PN (≥4PN)
    assert out["eps_h"][-1] > 0                                             # modos truncados irradian de menos


def test_D2_epsilon_h_detects_an_amplitude_error():
    """ε_h es el test independiente: una amplitud mal normalizada por k debe dar ε_h ≈ 1 - k^2 (1 - ε_h,base)."""
    d = mk.run_mock_rol1(rr_scheme="T1", dt=0.5)
    cfg = eng.Rol2Config()
    base = eng.compute_rol2(d, cfg)
    orig = pnc.mode_table
    def scaled(*a, **k):
        l, m, C, CL = orig(*a, **k)
        return l, m, 1.05*C, 1.05*CL
    pnc.mode_table = scaled
    try:
        bad = eng.compute_rol2(d, cfg)
    finally:
        pnc.mode_table = orig
    i = len(d["t"])//4
    expected = 1 - 1.05**2*(1 - base["eps_h"][i])
    assert abs(bad["eps_h"][i] - expected) < 1e-3
    assert abs(bad["eps_h"][i]) > 0.08                  # un 5 % de error en h NO pasa desapercibido


# ------------------------------------------------------------------ E
def test_E1_output_contract():
    out = eng.compute_rol2(mk.run_mock_rol1(dt=0.5), eng.Rol2Config(iota=0.7))
    t = out["t"]
    assert t[-1] == 0.0
    assert np.max(np.abs(np.diff(t) - 1/4096)) < 1e-15
    for k in ("h_plus", "h_cross", "f_gw"):
        assert np.all(np.isfinite(out[k])) and len(out[k]) == len(t)
    assert np.all(np.diff(out["f_gw"]) > 0)
    assert 19.9 < out["f_gw"][0] < 20.1 and 60 < out["f_gw"][-1] < 63        # 20 Hz -> ISCO (~62 Hz para 70.8 Msun)
    assert abs(out["h_plus"][0]) < 1e-3*np.abs(out["h_plus"]).max()          # taper al inicio
    assert 1e-22 < np.abs(out["h_plus"]).max() < 5e-21                       # orden de magnitud a 410 Mpc
    out2 = eng.compute_rol2(mk.run_mock_rol1(dt=0.5), eng.Rol2Config(iota=0.7, DL_Mpc=820.0))
    assert abs(np.abs(out2["h_plus"]).max()/np.abs(out["h_plus"]).max() - 0.5) < 1e-9   # h ∝ 1/D_L


def test_E2_rol1_validation():
    good = mk.run_mock_rol1(dt=0.5)
    assert rio.validate_rol1(good)
    bad = dict(good); bad["phi"] = np.mod(good["phi"], 2*pi)
    for b, msg in ((bad, "monótona"), (dict(good, x_pn=good["x_pn"][::-1]), "x_pn")):
        try:
            rio.validate_rol1(b); raise AssertionError("debía fallar")
        except ValueError as e:
            assert msg in str(e)
    nox = {k: v for k, v in good.items() if k not in ("x_pn", "phi")}          # derivación desde r, v
    rec = rio.normalize_rol1(nox)
    assert np.max(np.abs(rec["x_pn"]/good["x_pn"] - 1)) < 5e-3                  # r=m/γ(x) con ω exacto => x recuperado


# ------------------------------------------------------------------ F
def _chirp_metrics(t, hp, hc, fa=25.0, fb=50.0, fref=40.0):
    """Métricas independientes de la alineación temporal (ι=0: h+ + i hx es circular)."""
    dt = t[1] - t[0]
    s = hp + 1j*hc
    amp = np.abs(s)
    ph = np.unwrap(np.angle(s))
    f = np.abs(np.gradient(ph, dt))/(2*pi)
    core = (f > 0.8*fa) & (f < 1.2*fb)
    i0 = np.argmax(core)
    mono = np.maximum.accumulate(f[core])
    tc, phc, ac = t[core], ph[core], amp[core]
    ta, tb = np.interp([fa, fb], mono, tc)
    return dict(dt_ab=tb - ta, dphi_ab=abs(np.interp(tb, tc, phc) - np.interp(ta, tc, phc)),
                A_ref=np.interp(np.interp(fref, mono, tc), tc, ac))


def compare_with_reference(mine, ref, tol_dt=5e-3, tol_dphi=0.3, tol_amp=0.02):
    m, r = _chirp_metrics(*mine), _chirp_metrics(*ref)
    res = dict(dt_rel=m["dt_ab"]/r["dt_ab"] - 1, dphi_abs=m["dphi_ab"] - r["dphi_ab"], amp_rel=m["A_ref"]/r["A_ref"] - 1)
    ok = abs(res["dt_rel"]) < tol_dt and abs(res["dphi_abs"]) < tol_dphi and abs(res["amp_rel"]) < tol_amp
    return ok, res


def test_F0_metrics_selfcheck():
    out = eng.compute_rol2(mk.run_mock_rol1(rr_scheme="T4", dt=0.5), eng.Rol2Config(iota=0.0, amp_pn=0))
    mine = (out["t"], out["h_plus"], out["h_cross"])
    ok, res = compare_with_reference(mine, mine)
    assert ok and all(abs(v) < 1e-12 for v in res.values())


def test_F1_vs_pycbc():
    """Compara contra el MISMO aproximante que usa el Rol 1 (TaylorT1 y TaylorT4 difieren ~2.6 rad entre 25-50 Hz).
    Con amplitude_order igual en ambos lados (0 = newtoniano). Umbrales PROVISIONALES."""
    try:
        from pycbc.waveform import get_td_waveform
    except Exception:
        raise SkipTest("PyCBC no instalado")
    m1, m2, DL = 39.2, 31.6, 410.0
    for scheme, approx in (("T4", "TaylorT4"), ("T1", "TaylorT1")):
        d = mk.run_mock_rol1(m1, m2, f_gw0=20.0, rr_scheme=scheme, pn_order=7, dt=0.25)
        for amp in (0,):
            out = eng.compute_rol2(d, eng.Rol2Config(DL_Mpc=DL, iota=0.0, amp_pn=amp, flux_pn=7))
            hp, hc = get_td_waveform(approximant=approx, mass1=m1, mass2=m2, f_lower=20.0, delta_t=1/4096.,
                                     distance=DL, inclination=0.0, coa_phase=0.0, phase_order=7,
                                     amplitude_order=amp)
            ref = (np.arange(len(hp))/4096.0, np.asarray(hp), np.asarray(hc))
            ok, res = compare_with_reference((out["t"], out["h_plus"], out["h_cross"]), ref)
            print(f"   [PyCBC {approx}, amplitude_order={amp}] dt_rel={res['dt_rel']:+.2e}  "
                  f"dphi={res['dphi_abs']:+.3f} rad  dA_rel={res['amp_rel']:+.2e}")
            assert ok, (approx, res)


# ------------------------------------------------------------------ G
def test_G1_paraview_export_smoke():
    try:
        import pyevtk  # noqa
    except Exception:
        raise SkipTest("pyevtk no instalado")
    import rol2_paraview as pv
    out = eng.compute_rol2(mk.run_mock_rol1(dt=0.5), eng.Rol2Config(iota=0.5))
    with tempfile.TemporaryDirectory() as d:
        w = pv.export_paraview(out, d, n_frames=30, n_wave_frames=6, wave_n=41, trail_points=300)
        for k in ("bodies_pvd", "wavefield_pvd", "trail_m1", "trail_m2", "csv"):
            assert os.path.exists(w[k]), k
        assert open(w["bodies_pvd"]).read().count("<DataSet") == 30
        assert len([f for f in os.listdir(os.path.join(d, "wavefield")) if f.endswith(".vti")]) == 6


# ------------------------------------------------------------------ runner
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    nfail = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except SkipTest as e:
            print(f"SKIP  {name}  ({e})")
        except Exception:
            nfail += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{len(tests)-nfail}/{len(tests)} sin fallos (SKIP cuenta como sin fallo)")
    sys.exit(1 if nfail else 0)
