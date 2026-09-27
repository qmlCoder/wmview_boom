"""把 模型/utils/special.py 里 sympy 现场推导的几何基函数固化成数值代码。

原模型在 import 时用 sympy 符号推导
    - 球贝塞尔零点 / 归一化系数（BesselBasisLayer、SphericalBasisLayer、torsionLayer）
    - 实球谐函数（m=0 的 sbf 基组；全部 m 的 tbf 基组）
再 lambdify 成 torch 函数。推理运行时不需要 sympy，所以这里把推导结果
固化成 runtime/basis_gen.py（纯 numpy 代码 + 数值系数表）。

用法（需要 sympy/scipy，用 miniconda 的 python 跑）：
    python tools/gen_basis.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import sympy as sym
from sympy.printing.numpy import NumPyPrinter
from scipy.optimize import brentq
from scipy.special import jv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runtime" / "basis_gen.py"

# ---------------- 以下函数逐行对应 模型/utils/special.py ----------------


def sph_harm_prefactor(k, m):
    return ((2 * k + 1) * math.factorial(k - abs(m)) /
            (4 * math.pi * math.factorial(k + abs(m))))**0.5


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * jv(n + 0.5, r)


def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i, ))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    return zerosj


def spherical_bessel_formulas(n):
    x = sym.symbols('x')
    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x)**i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]
    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def associated_legendre_polynomials(k, zero_m_only=True):
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] -
                                        (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))
    return P_l_m


def real_sph_harm(k, zero_m_only=True, spherical_coordinates=True):
    if not zero_m_only:
        x, y = sym.symbols('x y')
        S_m = [sym.sympify(0)]
        C_m = [sym.sympify(1)]
        for i in range(1, k):
            S_m.append(x * S_m[i - 1] + y * C_m[i - 1])
            C_m.append(x * C_m[i - 1] - y * S_m[i - 1])

    P_l_m = associated_legendre_polynomials(k, zero_m_only)

    if spherical_coordinates:
        theta = sym.symbols('theta')
        z_sym = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if isinstance(P_l_m[i][j], sym.Expr):
                    P_l_m[i][j] = P_l_m[i][j].subs(z_sym, sym.cos(theta))

        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                if isinstance(S_m[i], sym.Expr) and S_m[i] != 0:
                    S_m[i] = S_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(
                        y, sym.sin(theta) * sym.sin(phi))
            for i in range(len(C_m)):
                if isinstance(C_m[i], sym.Expr) and C_m[i] != 0:
                    C_m[i] = C_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(
                        y, sym.sin(theta) * sym.sin(phi))

    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(k)]

    for i in range(k):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


# ---------------- 生成数值代码 ----------------

MODULES = {'sin': sym.sin, 'cos': sym.cos}  # 与原代码一致（lambdify 用 torch.sin/cos）


_PRINTER = NumPyPrinter()


def _pycode(expr) -> str:
    """sympy 表达式 -> 纯 numpy 代码（把 numpy.xxx 缩写成 np.xxx）。"""
    return _PRINTER.doprint(expr).replace('numpy.', 'np.')


def _f32(v) -> float:
    """复现原代码：零点以 float32 存储后再参与符号代入。"""
    return float(np.float32(v))


def bessel_pycode(num_spherical, num_radial):
    """返回 (zeros, norms, [每个 (l, n) 的 pycode 表达式])。"""
    zeros = Jn_zeros(num_spherical, num_radial)
    forms = spherical_bessel_formulas(num_spherical)
    x = sym.symbols('x')
    codes = []
    norms = np.zeros((num_spherical, num_radial))
    for l in range(num_spherical):
        for n in range(num_radial):
            z = _f32(zeros[l, n])
            norm = 0.5 * Jn(zeros[l, n], l + 1) ** 2
            norm = 1.0 / float(norm) ** 0.5
            norms[l, n] = norm
            expr = sym.simplify(sym.Float(repr(norm)) *
                                forms[l].subs(x, sym.Float(repr(z)) * x))
            codes.append(_pycode(expr))
    return zeros, norms, codes


def sph_pycode(num_spherical, zero_m_only):
    """球谐部分：返回按 special.py 里 sph_funcs 追加顺序排列的 pycode 列表。

    zero_m_only=True  → 只有 m=0，共 num_spherical 个（sbf 用，只依赖 theta）
    zero_m_only=False → l=0 一个 + l=1 三个 + l=2 五个（tbf 用，依赖 theta/phi）
    """
    theta, phi = sym.symbols('theta phi')
    forms = real_sph_harm(num_spherical, zero_m_only=zero_m_only)
    codes = []
    for l in range(num_spherical):
        if zero_m_only:
            expr = forms[l][0]
            if l == 0:
                # 原代码：sph1 = lambdify(...)(0)，再把 theta 全部吃掉 → 常数
                const = float(sym.lambdify([theta], expr, MODULES)(0.0))
                codes.append(repr(const))
            else:
                codes.append(_pycode(expr))
        else:
            if l == 0:
                const = float(sym.lambdify([theta, phi], forms[l][0], MODULES)(0.0, 0.0))
                codes.append(repr(const))
            else:
                for m in range(-l, l + 1):
                    codes.append(_pycode(forms[l][m + l]))
    return codes


def main():
    num_spherical, num_radial = 3, 6
    zeros, norms, bessel_codes = bessel_pycode(num_spherical, num_radial)
    sbf_codes = sph_pycode(num_spherical, zero_m_only=True)
    tbf_codes = sph_pycode(num_spherical, zero_m_only=False)
    assert len(sbf_codes) == num_spherical, len(sbf_codes)
    assert len(tbf_codes) == 1 + 3 + 5, len(tbf_codes)

    def arr(a):
        return np.array2string(np.asarray(a, dtype=np.float64), precision=17,
                               separator=', ', threshold=10 ** 9)

    lines = [
        '"""由 tools/gen_basis.py 自动生成，请勿手改。',
        '',
        '内容 = 模型/utils/special.py 中 sympy 现场推导的几何基函数（数值化固化）。',
        '"""',
        '',
        'import numpy as np',
        '',
        f'NUM_SPHERICAL = {num_spherical}',
        f'NUM_RADIAL = {num_radial}',
        '',
        '# 球贝塞尔零点（shape: [NUM_SPHERICAL, NUM_RADIAL]）',
        f'BESSEL_ZEROS = np.array({arr(zeros)}, dtype=np.float64)',
        '',
        '# 归一化系数（shape: [NUM_SPHERICAL, NUM_RADIAL]）',
        f'BESSEL_NORM = np.array({arr(norms)}, dtype=np.float64)',
        '',
        '',
        'def sbf_radial(rs):',
        '    """(M,) 归一化距离 -> (M, NUM_SPHERICAL*NUM_RADIAL) 球贝塞尔基组。"""',
        '    rs = np.asarray(rs, dtype=np.float64)',
        f'    out = np.empty((rs.shape[0], {num_spherical * num_radial}), dtype=np.float64)',
        '    x = rs',
    ]
    for idx, code in enumerate(bessel_codes):
        lines.append(f'    out[:, {idx}] = {code}')
    lines += [
        '    return out',
        '',
        '',
        'def sph_sbf(theta):',
        '    """(M,) 键角 -> (M, NUM_SPHERICAL) 实球谐（m=0）。"""',
        '    theta = np.asarray(theta, dtype=np.float64)',
        f'    out = np.empty((theta.shape[0], {len(sbf_codes)}), dtype=np.float64)',
    ]
    for idx, code in enumerate(sbf_codes):
        lines.append(f'    out[:, {idx}] = {code}')
    lines += [
        '    return out',
        '',
        '',
        'def sph_tbf(theta, phi):',
        '    """(M,) 键角 + (M,) 二面角 -> (M, 9) 实球谐（l=0..2，全部 m）。"""',
        '    theta = np.asarray(theta, dtype=np.float64)',
        '    phi = np.asarray(phi, dtype=np.float64)',
        f'    out = np.empty((theta.shape[0], {len(tbf_codes)}), dtype=np.float64)',
    ]
    for idx, code in enumerate(tbf_codes):
        lines.append(f'    out[:, {idx}] = {code}')
    lines += [
        '    return out',
        '',
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'written: {OUT}')
    print('bessel zeros:\n', zeros)
    print('sph codes (sbf):', sbf_codes)
    print('sph codes (tbf):', tbf_codes)


if __name__ == '__main__':
    main()

