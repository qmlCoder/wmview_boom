"""拟合「ρ + ΔHf -> Q / D / p」的经验关系（模型只输出 ρ 和 ΔHf）。

动机（见 模型/README.md §5）：原项目只训了 best_density / best_eof 两个模型，
Q(爆热)、D(爆速)、p(爆压) 不是网络输出。本工具在随包数据集 data/final_123679.csv
上做**线性回归**，把这三个量写成 (分子式元素计数, ρ, ΔHf, 氧平衡产物) 的经验函数，
供 runtime/perf.py 使用。

产物里明确标注是「经验估算」，不是模型输出；README 里记录了留出集验证误差。

用法：
    python tools/fit_perf.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "模型" / "data" / "final_123679.csv"
CSV_ALT = ROOT / "model" / "data" / "final_123679.csv"
OUT = ROOT / "runtime" / "weights" / "perf_coeff.json"

PAT = re.compile(r"([A-Z][a-z]?)(\d*)")
MW_ATOM = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}

FEATURES = [
    "bias", "C", "H", "N", "O", "rho", "hof",
    "nCO2", "nH2O", "nCO", "nO2", "nN2", "ngas", "Mgas",
]
TARGETS = {"Q": "Q(cal/g)", "D": "D(Km/s)", "p": "p(GPa)"}


def parse_formula(text: str):
    counts = {}
    for sym, n in PAT.findall(text):
        counts[sym] = counts.get(sym, 0) + (int(n) if n else 1)
    return counts


def products(c, h, n, o):
    """标准 CHNO 氧平衡产物规则（H 先成水，剩余氧给碳，最后才出 O2）。"""
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


def build():
    src = CSV if CSV.exists() else CSV_ALT
    df = pd.read_csv(src)
    rec = np.array([parse_formula(f) for f in df["Formula"]], dtype=object)
    c = np.array([r.get("C", 0) for r in rec], dtype=float)
    h = np.array([r.get("H", 0) for r in rec], dtype=float)
    n = np.array([r.get("N", 0) for r in rec], dtype=float)
    o = np.array([r.get("O", 0) for r in rec], dtype=float)
    rho = df["density(g/cm3)"].to_numpy(float)
    hof = df["solid_HOF(kcal/mol)"].to_numpy(float)
    nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas = products(c, h, n, o)
    X = np.column_stack([np.ones_like(c), c, h, n, o, rho, hof,
                         nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas])
    y = {k: df[col].to_numpy(float) for k, col in TARGETS.items()}
    return X, y, len(df)


def main():
    X, y, n_all = build()
    rng = np.random.default_rng(641)
    idx = rng.permutation(n_all)
    n_val = n_all // 5
    val, tr = idx[:n_val], idx[n_val:]
    out = {
        "desc": "由 tools/fit_perf.py 在随包数据集上拟合的经验关系（非模型输出）",
        "source_csv": "模型/data/final_123679.csv",
        "features": FEATURES,
        "products_rule": "CHNO 氧平衡：H→H2O，余氧→CO2，再余氧→O2，N→N2",
        "targets": {},
    }
    for name, col in TARGETS.items():
        truth = y[name][val]
        best = None
        for mode in ('linear', 'log'):
            if mode == 'log' and np.any(y[name][tr] <= 0):
                continue
            t = y[name][tr] if mode == 'linear' else np.log(y[name][tr])
            coef, *_ = np.linalg.lstsq(X[tr], t, rcond=None)
            pred = X[val] @ coef
            if mode == 'log':
                pred = np.exp(pred)
            mre = float(np.median(np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-9)))
            cand = {
                "mode": mode,
                "coef": [float(v) for v in coef],
                "holdout": {
                    "mae": float(np.mean(np.abs(pred - truth))),
                    "median_rel": mre,
                    "r2": float(1 - np.sum((pred - truth) ** 2)
                                / np.sum((truth - truth.mean()) ** 2)),
                    "n": int(n_val),
                },
            }
            if best is None or mre < best["holdout"]["median_rel"]:
                best = cand
        c_full = np.array(best["coef"])
        pred = X[val] @ c_full
        if best["mode"] == 'log':
            pred = np.exp(pred)
        mae = float(np.mean(np.abs(pred - truth)))
        mre = float(np.median(np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-9)))
        r2 = float(1 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean()) ** 2))
        out["targets"][name] = {
            "column": col,
            "mode": best["mode"],
            "coef": [float(v) for v in c_full],
            "holdout": {"mae": mae, "median_rel": mre, "r2": r2, "n": int(n_val)},
        }
        print(f"{name:2s} [{best['mode']:6s}]: holdout MAE={mae:.4f} "
              f"中位相对误差={mre * 100:.2f}% R2={r2:.5f}")
    out["note"] = ("Q/D/p 只由 (分子式, 密度, 生成焓) 经验拟合，同分异构体差异无法体现；"
                   "仅作量级参考，不是模型输出。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written:", OUT)


if __name__ == "__main__":
    main()
