# Rol 2 — h₊(t), hₓ(t) y balance energético (hasta 3.5PN)

| Archivo | Qué es |
|---|---|
| `pn_coeffs.py` | Tablas de coeficientes (E, F, ẋ de TaylorT4, modos h_ℓm, armónicos _{-2}Y_ℓm). NumPy puro. |
| `rol2_engine.py` | **Kernels `@njit(cache=True)`** y `compute_rol2(rol1, cfg)`. |
| `rol2_io.py` | Contrato de entrada (validación), `to_physical()` (única fuente de unidades), escritura de salidas. |
| `rol2_paraview.py` | Exportador pyevtk → `.pvd` + `.vtu` + `.vti` + CSV. Python puro. |
| `mock_rol1.py` | **Sustituto** del Rol 1 (RK4, T1/T4/0PN). No es su código; ejemplo ejecutable del contrato. |
| `run_rol2.py` | Driver: `python run_rol2.py --rol1 rol1_output.npz --out rol2_out --paraview --plot` |
| `test_rol2.py` | Validación (A–G). `python test_rol2.py` |

## Contrato
**Rol 1 → Rol 2** (`rol1_output.npz`, float64): `t` (G m/c³, creciente), `x_pn`, `phi` (acumulada, sin envolver),
`r1,r2,v1,v2` (solo visualización), escalares `m1_Msun, m2_Msun` (marco detector) y `rr_scheme` ('T1' | 'T4').
Unidades G = c = m = 1. ≥ 50 pasos/órbita. Arranque en f_gw ≈ 20 Hz, corte en x = 1/6.

**Rol 2 → Rol 3** (`h_sim.npz`): `t` [s] (dt = 1/4096 exacto, t = 0 al final), `h_plus`, `h_cross` (a D_L, ι fiduciales,
con taper, **sin filtrar**), `f_gw` [Hz], más `meta_json` y escalares. Rol 3 debe dejar libres Δt, φ_c y un factor de
amplitud, y aplicar a esta señal el mismo pasa-banda/whitening que a los datos de LIGO.

**Todo el equipo** (`energy_balance.npz`): t, x, E_orb, E_rad^F, E_rad^h, eps_F, eps_h, M_sys(t) [M☉].

## Órdenes PN implementados
- Energía E(x): 3PN (la parte conservativa no tiene 3.5PN). Flujo F(x), ẋ: **3.5PN**.
- Amplitud: h₂₂ a 3PN; h₂₁, h₃₃, h₃₂, h₃₁ a 2.5PN; h₄ₘ, h₅ₘ solo lo listado en `pn_coeffs.py`. **No** está el 3.5PN de la amplitud
  ni los modos (5,4), (5,2), ℓ ≥ 6, m = 0 (memoria).
- Fase: ψ = φ − 3x^{3/2}(1 − νx/2) ln(x/x₀). Cambiar x₀ equivale a un **corrimiento temporal** (δψ ∝ ω).

## Validación (ver test_rol2.py)
- ẋ(TaylorT4) derivado de mis E y F = literatura, 1e-14, varios ν.
- Σ flujos de los modos = F(x) **exacto hasta 3PN** (incluye log, γ_E, π²) → normalización y módulos de los modos.
- Cuadrupolo de Einstein sobre la cinemática (r,v,a) vs modo (2,2) a 0PN: 1.8 % rms (residuo = reacción de radiación 2.5PN).
- ε_F con TaylorT1: ≈ 4e-9 (solo numérico, es identidad por construcción). Con TaylorT4: −1 % (x=0.08) → −8.5 % (ISCO): mide la truncación de ẋ.
- Desajuste INSTANTÁNEO flux_h/F − 1 (T1 o T4, no depende de la dinámica): −0.5 % (x=0.08) → −9 % (ISCO), escala ~x^3.9 = truncamiento PN.
  El ε_h ACUMULADO (T1) es un promedio ponderado desde x₀: 0.5 % → 3.1 %, escala ~x^2.3. Un error de normalización sería constante, no esto.
- **TaylorT1 y T4 al mismo orden difieren ~2.6 rad entre 25 y 50 Hz.** Comparar con PyCBC solo contra el aproximante de Rol 1.

## Pendiente de verificar en una máquina con Numba, pyevtk y PyCBC
Los kernels no se compilaron con Numba aquí (se ejecutó el mismo código en Python puro); el exportador se probó contra un stub
de pyevtk; la comparación con PyCBC no se ejecutó (umbrales provisionales). Orden sugerido:
`python test_rol2.py` → `NUMBA_DISABLE_JIT=1 python test_rol2.py` (deben coincidir) → `python run_rol2.py --mock --paraview --plot`.
