"""
run_rol2.py -- Driver del Rol 2:  salida del Rol 1  ->  h_sim.npz (Rol 3), energy_balance.npz, ParaView.

Uso
    python run_rol2.py --rol1 rol1_output.npz --out rol2_out --DL 410 --iota 0.0 --paraview --plot
    python run_rol2.py --mock --out rol2_out --paraview --plot     # prueba sin el código del Rol 1

Salidas en --out:
    h_sim.npz           t[s], h_plus, h_cross, f_gw[Hz] + meta (para Rol 3; 4096 Hz exacto, t=0 al final)
    energy_balance.npz  t, x, E_orb, E_rad^F, E_rad^h, eps_F, eps_h, M_sys(t) [M_sun], flujos
    diagnostics.png     (--plot)     paraview/  (--paraview)
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rol2_io as rio
import rol2_engine as eng


def plot_diagnostics(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = out["t"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    ax[0, 0].plot(t, out["h_plus"]*1e21, lw=0.9, label=r"$h_+$")
    ax[0, 0].plot(t, out["h_cross"]*1e21, lw=0.9, label=r"$h_\times$")
    ax[0, 0].set(xlabel="t [s]  (0 = fin de la integración)", ylabel=r"$h\ [10^{-21}]$",
                 title=f"Forma de onda ({out['meta']['pn_order_amp']/2:g}PN amplitud)")
    ax[0, 0].legend()
    ax[0, 1].plot(t, out["f_gw"]); ax[0, 1].set(xlabel="t [s]", ylabel=r"$f_{gw}$ [Hz]", title="Chirp")
    ax[1, 0].plot(t, out["E_rad_F_Msun"], label=r"$E_{rad}^F=\int F\,dt$")
    ax[1, 0].plot(t, out["E_rad_h_Msun"], "--", label=r"$E_{rad}^h$ (desde $\dot h_{\ell m}$)")
    ax[1, 0].plot(t, out["M_sys_Msun"][0] - out["M_sys_Msun"], ":", label=r"$M_{sys}(t_0)-M_{sys}(t)$")
    ax[1, 0].set(xlabel="t [s]", ylabel=r"$[M_\odot c^2]$", title="Balance energético"); ax[1, 0].legend()
    ax[1, 1].semilogy(t, np.abs(out["eps_F"]) + 1e-16, label=r"$|\epsilon_F|$ (integración)")
    ax[1, 1].semilogy(t, np.abs(out["eps_h"]) + 1e-16, label=r"$|\epsilon_h|$ (truncamiento PN)")
    ax[1, 1].set(xlabel="t [s]", ylabel="residuo relativo", title=r"$\epsilon=1-E_{rad}/\Delta E_{orb}$"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rol1", help="rol1_output.npz del Rol 1")
    ap.add_argument("--mock", action="store_true", help="usar mock_rol1 (TaylorT1) en lugar de un archivo")
    ap.add_argument("--out", default="rol2_out")
    ap.add_argument("--DL", type=float, default=410.0, help="distancia de luminosidad [Mpc]")
    ap.add_argument("--iota", type=float, default=0.0, help="inclinación [rad]")
    ap.add_argument("--phic", type=float, default=0.0, help="fase ψ en t=0 [rad]")
    ap.add_argument("--flux-pn", type=int, default=7); ap.add_argument("--energy-pn", type=int, default=6)
    ap.add_argument("--amp-pn", type=int, default=6)
    ap.add_argument("--no-tail", action="store_true")
    ap.add_argument("--paraview", action="store_true"); ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--wave-frames", type=int, default=150)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)

    if a.mock:
        import mock_rol1
        rol1 = mock_rol1.run_mock_rol1()
    elif a.rol1:
        rol1 = rio.load_rol1(a.rol1)
    else:
        ap.error("indica --rol1 ARCHIVO o --mock")
    rio.validate_rol1(rol1)
    os.makedirs(a.out, exist_ok=True)

    cfg = eng.Rol2Config(DL_Mpc=a.DL, iota=a.iota, phi_c=a.phic, flux_pn=a.flux_pn, energy_pn=a.energy_pn,
                         amp_pn=a.amp_pn, include_tail=not a.no_tail)
    out = eng.compute_rol2(rol1, cfg)
    out["meta"]["rol1_rr_scheme"] = str(rol1.get("rr_scheme", "desconocido"))
    rio.save_h_sim(os.path.join(a.out, "h_sim.npz"), out, out["meta"])
    rio.save_energy_balance(os.path.join(a.out, "energy_balance.npz"), out)

    m = out["meta"]
    print(f"numba: {eng.HAVE_NUMBA} | m1={m['m1_Msun']} m2={m['m2_Msun']} Msun, nu={m['nu']:.5f}, DL={m['DL_Mpc']} Mpc")
    print(f"señal: {-out['t'][0]:.4f} s, {len(out['t'])} muestras a {m['fs']:.0f} Hz, "
          f"f_gw {m['f_gw_start']:.1f} -> {m['f_gw_end']:.1f} Hz, x {m['x_start']:.4f} -> {m['x_max']:.4f}")
    print(f"max|h+|={np.abs(out['h_plus']).max():.3e}  max|hx|={np.abs(out['h_cross']).max():.3e}")
    print(f"E_rad (F)={out['E_rad_F_Msun'][-1]:.4f}  E_rad (h)={out['E_rad_h_Msun'][-1]:.4f}  "
          f"M(t0)-M(t_end)={out['M_sys_Msun'][0]-out['M_sys_Msun'][-1]:.4f}  [Msun c^2]")
    print(f"eps_F final={out['eps_F'][-1]:+.2e} (numérico)   eps_h final={out['eps_h'][-1]:+.2e} (truncamiento PN)")
    if a.plot:
        plot_diagnostics(out, os.path.join(a.out, "diagnostics.png"))
    if a.paraview:
        import rol2_paraview as pv
        w = pv.export_paraview(out, os.path.join(a.out, "paraview"), n_frames=a.frames, n_wave_frames=a.wave_frames)
        print("ParaView:", ", ".join(os.path.relpath(p, a.out) for p in w.values()))
    return out


if __name__ == "__main__":
    main()
