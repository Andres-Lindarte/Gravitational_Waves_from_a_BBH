"""
pn_coeffs.py  --  Tablas de coeficientes post-newtonianos (Rol 2)
=================================================================
NumPy puro (sin Numba): se ejecuta UNA vez por corrida y entrega arreglos
que los kernels @njit de `rol2_engine.py` solo evalúan.

Convenciones (G = c = 1, m = m1 + m2 = 1)
    nu    = m1 m2 / m^2           (razón de masas simétrica)
    delta = (m1 - m2) / m         (>= 0 si m1 >= m2)
    x     = (m*omega)^(2/3),   y = sqrt(x)
    L     = ln(16 x)               (único logaritmo que aparece hasta 3.5PN)

`pn_order` se mide en SEMIÓRDENES: 0=0PN, 1=0.5PN, ..., 6=3PN, 7=3.5PN.
Una serie en y = sqrt(x) se guarda como arreglo de longitud 8 (potencias y^0..y^7).

Fuentes: Blanchet, Living Rev. Relativ. 17, 2 (2014) y Blanchet-Faye-Iyer-Sinha,
CQG 25, 165003 (2008) [arXiv:0802.1249]. La verificación numérica está en
test_rol2.py (flujo de los modos vs F(x); ẋ de TaylorT4 vs literatura).
"""
from math import pi, sqrt, log, factorial, cos, sin
import numpy as np

EULER_GAMMA = 0.5772156649015329
NK = 8                                   # potencias y^0 ... y^7


# ----------------------------------------------------------------------------
# 1. Energía de ligadura y flujo (órbitas cuasi-circulares)
# ----------------------------------------------------------------------------
def energy_coeffs(nu, pn_order=6):
    """E(x) = -(nu x / 2) * sum_n e[n] x^n   [en unidades de m c^2, m=1].
    Devuelve e[0..3] (potencias ENTERAS de x). No hay términos semienteros
    en la parte conservativa hasta 3PN; `pn_order` recorta 2n <= pn_order."""
    e = np.array([
        1.0,
        -(3/4 + nu/12),
        -(27/8 - 19*nu/8 + nu**2/24),
        -(675/64 - (34445/576 - 205*pi**2/96)*nu + 155*nu**2/96 + 35*nu**3/5184),
    ])
    for n in range(4):
        if 2*n > pn_order:
            e[n] = 0.0
    return e


def flux_coeffs(nu, pn_order=7):
    """F(x) = (32/5) nu^2 x^5 * sum_k (f[k] + fl[k] L) y^k ,  L = ln(16x).
    Devuelve (f, fl), ambos de longitud 8."""
    g = EULER_GAMMA
    f = np.zeros(NK)
    fl = np.zeros(NK)
    f[0] = 1.0
    f[2] = -(1247/336 + 35*nu/12)
    f[3] = 4*pi
    f[4] = -44711/9072 + 9271*nu/504 + 65*nu**2/18
    f[5] = -(8191/672 + 583*nu/24)*pi
    f[6] = (6643739519/69854400 + 16*pi**2/3 - 1712*g/105
            + (-134543/7776 + 41*pi**2/48)*nu - 94403*nu**2/3024 - 775*nu**3/324)
    fl[6] = -856/105
    f[7] = (-16285/504 + 214745*nu/1728 + 193385*nu**2/3024)*pi
    f[pn_order+1:] = 0.0
    fl[pn_order+1:] = 0.0
    return f, fl


def _series_div(a, b):
    """a(y)/b(y) para series de longitud NK con b[0] != 0."""
    c = np.zeros(NK)
    for n in range(NK):
        c[n] = (a[n] - sum(b[k]*c[n-k] for k in range(1, n+1))) / b[0]
    return c


def dxdt_t4_coeffs(nu, pn_order=7):
    """TaylorT4:  dx/dt = (64/5) nu x^5 * sum_k (t[k] + tl[k] L) y^k  con
    t = f / e', e'(x) = sum_n (n+1) e[n] x^n  (se hace la división de series).
    Sirve al Rol 1 como alternativa a TaylorT1 (dx/dt = -F/E' sin truncar)."""
    f, fl = flux_coeffs(nu, pn_order)
    e = energy_coeffs(nu, pn_order)
    ep = np.zeros(NK)
    for n in range(4):
        ep[2*n] = (n+1)*e[n]            # x^n = y^(2n)
    t = _series_div(f, ep)
    # el log solo aparece en y^6 y multiplica a 1/e'(0)=1 (los productos con
    # y^2 caen en y^8, fuera de 3.5PN):
    tl = np.zeros(NK)
    tl[6] = fl[6] / ep[0]
    t[pn_order+1:] = 0.0
    tl[pn_order+1:] = 0.0
    return t, tl


# ----------------------------------------------------------------------------
# 2. Modos h_lm:   h_lm = (2 m nu x / R) sqrt(16 pi/5) * H_lm(x) * exp(-i m psi)
#    H_lm(x) = sum_k (C[k] + CL[k] L) y^k      (C, CL complejos; incluye el prefactor)
# ----------------------------------------------------------------------------
MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (3, 1),
         (4, 4), (4, 3), (4, 2), (4, 1),
         (5, 5), (5, 3), (5, 1)]
# Omitidos a propósito (aportan < 1e-3 de h22 en toda la inspiral): (5,4), (5,2), ℓ>=6 y m=0 (memoria DC).


def mode_table(nu, delta, amp_pn=6):
    """Devuelve (ell[NM], m[NM], C[NM,NK], CL[NM,NK]) con H_lm truncado a y^k, k<=amp_pn.
    h22 llega a 3PN (k=6); los modos subdominantes se dan hasta y^5 (2.5PN) como máximo."""
    g = EULER_GAMMA
    d = delta
    I = 1j
    tab = {}

    tab[(2, 2)] = (1.0, {
        0: 1.0,
        2: -107/42 + 55*nu/42,
        3: 2*pi,
        4: -2173/1512 - 1069*nu/216 + 2047*nu**2/1512,
        5: -107*pi/21 + (34*pi/21 - 24*I)*nu,
        6: (27027409/646800 - 856*g/105 + 428*I*pi/105 + 2*pi**2/3
            + (-278185/33264 + 41*pi**2/96)*nu - 20261*nu**2/2772 + 114635*nu**3/99792)},
        {6: -428/105})
    tab[(2, 1)] = (I*d/3, {
        1: 1.0, 3: -17/28 + 5*nu/7,
        4: pi + I*(-1/2 - 2*log(2)),
        5: -43/126 - 509*nu/126 + 79*nu**2/168}, {})
    tab[(3, 3)] = (-0.75*I*sqrt(15/14)*d, {
        1: 1.0, 3: -4 + 2*nu,
        4: 3*pi + I*(-21/5 + 6*log(3/2)),
        5: 123/110 - 1838*nu/165 + 887*nu**2/330}, {})
    tab[(3, 2)] = (sqrt(5/7)/3, {
        2: 1 - 3*nu, 4: -193/90 + 145*nu/18 - 73*nu**2/18,
        5: 2*pi*(1 - 3*nu) + I*(-3 + 66*nu/5)}, {})
    tab[(3, 1)] = (I*d/(12*sqrt(14)), {
        1: 1.0, 3: -8/3 - 2*nu/3,
        4: pi + I*(-7/5 - 2*log(2)),
        5: 607/198 - 136*nu/99 - 247*nu**2/198}, {})
    tab[(4, 4)] = (-(8/9)*sqrt(5/7), {
        2: 1 - 3*nu, 4: -593/110 + 1273*nu/66 - 175*nu**2/22}, {})
    tab[(4, 3)] = (-9*I*d/(4*sqrt(70)), {3: 1 - 2*nu}, {})
    tab[(4, 2)] = (sqrt(5)/63, {
        2: 1 - 3*nu, 4: -437/110 + 805*nu/66 - 19*nu**2/22}, {})
    tab[(4, 1)] = (I*d/(84*sqrt(10)), {3: 1 - 2*nu}, {})
    tab[(5, 5)] = (625*I*d/(96*sqrt(66)), {3: 1 - 2*nu}, {})
    tab[(5, 3)] = (-9*I*d/32*sqrt(3/110), {3: 1 - 2*nu}, {})   # módulo verificado (multipolos + flujo 3PN)
    tab[(5, 1)] = (I*d/(288*sqrt(385)), {3: 1 - 2*nu}, {})

    NM = len(MODES)
    ell = np.zeros(NM, dtype=np.int64)
    mm = np.zeros(NM, dtype=np.int64)
    C = np.zeros((NM, NK), dtype=np.complex128)
    CL = np.zeros((NM, NK), dtype=np.complex128)
    for j, (l, m) in enumerate(MODES):
        pref, ser, serl = tab[(l, m)]
        ell[j], mm[j] = l, m
        for k, v in ser.items():
            if k <= amp_pn:
                C[j, k] = pref * v
        for k, v in serl.items():
            if k <= amp_pn:
                CL[j, k] = pref * v
    return ell, mm, C, CL


# ----------------------------------------------------------------------------
# 3. Armónicos esféricos de peso de espín -2  (BFIS08, Eq. 2.4)
# ----------------------------------------------------------------------------
def swsh_m2(ell, m, theta, phi):
    """_{-2}Y_lm(theta, phi) = sqrt((2l+1)/4pi) d_2^{lm}(theta) e^{i m phi}."""
    k1 = max(0, m - 2)
    k2 = min(ell + m, ell - 2)
    s = 0.0
    num = sqrt(factorial(ell+m) * factorial(ell-m) * factorial(ell+2) * factorial(ell-2))
    ch, sh = cos(theta/2), sin(theta/2)
    for k in range(k1, k2 + 1):
        den = factorial(k - m + 2) * factorial(ell + m - k) * factorial(ell - k - 2)
        s += ((-1)**k / factorial(k)) * num / den * ch**(2*ell + m - 2*k - 2) * sh**(2*k - m + 2)
    return sqrt((2*ell + 1) / (4*pi)) * s * np.exp(1j * m * phi)


def angular_factors(ell, mm, iota, phi_obs):
    """Para cada modo (l, m>0): Yp = Y_lm(iota, phi_obs) y
    Ym = (-1)^l * Y_{l,-m}(iota, phi_obs), de modo que
        h+ - i hx = sum_modos [ h_lm * Yp + conj(h_lm) * Ym ]."""
    Yp = np.zeros(len(ell), dtype=np.complex128)
    Ym = np.zeros(len(ell), dtype=np.complex128)
    for j in range(len(ell)):
        Yp[j] = swsh_m2(int(ell[j]), int(mm[j]), iota, phi_obs)
        Ym[j] = (-1)**int(ell[j]) * swsh_m2(int(ell[j]), -int(mm[j]), iota, phi_obs)
    return Yp, Ym


# ----------------------------------------------------------------------------
# 4. Relación armónica separación-frecuencia (solo para reconstruir r1, r2)
# ----------------------------------------------------------------------------
def gamma_of_x(x, nu):
    """gamma = G m/(r c^2) en coordenadas armónicas, a 2PN (BFIS08 Eq. 6.6, sin el término 3PN)."""
    return x * (1.0 + x * (1.0 - nu/3.0) + x**2 * (1.0 - 65.0*nu/12.0))


def dgamma_dx(x, nu):
    return 1.0 + 2.0 * x * (1.0 - nu/3.0) + 3.0 * x**2 * (1.0 - 65.0*nu/12.0)
