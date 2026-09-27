"""爆轰性能的经验估算（Q / D / p）。

模型只输出密度 ρ 与固体生成焓 ΔHf；Q(爆热)、D(爆速)、p(爆压) 不是网络输出
（见 模型/README.md §5）。这里用 tools/fit_perf.py 在随包数据集
data/final_123679.csv 上拟合的线性关系做量级估算，**不是模型输出**。

留出集（20%）验证误差见 weights/perf_coeff.json：
    Q 中位相对误差 ~1%，D ~4%，p ~5%。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_coeff(weights_dir):
    path = Path(weights_dir) / 'perf_coeff.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def products(c, h, n, o):
    """标准 CHNO 氧平衡产物：H→H2O，余氧→CO2，再余氧→O2，N→N2。

    返回 (nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas)。
    """
    nH2O = h / 2.0
    rem = o - nH2O
    nCO2 = np.where(rem > 0, np.minimum(c, np.maximum(rem, 0) / 2.0), 0.0)
    rem2 = np.where(rem > 0, np.maximum(rem, 0) - 2 * nCO2, 0.0)
    nCO = np.where(rem > 0, np.maximum(c - nCO2, 0), 0.0)
    nO2 = rem2 / 2.0
    nN2 = n / 2.0
    ngas = nH2O + nCO2 + nCO + nO2 + nN2
    mass = nH2O * 18.015 + nCO2 * 44.01 + nCO * 28.01 + nO2 * 32.0 + nN2 * 28.014
    Mgas = np.where(ngas > 0, mass / np.maximum(ngas, 1e-9), 0.0)
    return nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas


def feature_vector(counts, rho, hof, coeff):
    c = float(counts.get('C', 0))
    h = float(counts.get('H', 0))
    n = float(counts.get('N', 0))
    o = float(counts.get('O', 0))
    nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas = products(c, h, n, o)
    values = {
        'bias': 1.0, 'C': c, 'H': h, 'N': n, 'O': o, 'rho': rho, 'hof': hof,
        'nCO2': float(nCO2), 'nH2O': float(nH2O), 'nCO': float(nCO), 'nO2': float(nO2),
        'nN2': float(nN2), 'ngas': float(ngas), 'Mgas': float(Mgas),
    }
    return np.array([values[k] for k in coeff['features']], dtype=np.float64), values


def estimate(counts, rho, hof, coeff):
    """返回 {'Q':…, 'D':…, 'p':…}（单位分别是 cal/g、km/s、GPa）。"""
    if coeff is None:
        return None
    x, _ = feature_vector(counts, rho, hof, coeff)
    out = {}
    for name, tgt in coeff['targets'].items():
        y = float(x @ np.asarray(tgt['coef'], dtype=np.float64))
        if tgt.get('mode') == 'log':
            y = float(np.exp(y))
        out[name] = y
    return out
