"""拟合「ρ + ΔHf -> Q / D / p」的经验关系（模型只输出 ρ 和 ΔHf）。

动机（见 模型/README.md §5）：原项目只训了 best_density / best_eof 两个模型，
Q(爆热)、D(爆速)、p(爆压) **不是网络输出**。本工具在随包数据集
模型/data/final_123679.csv 上做回归，产出两套系数供 Rust 推理核使用：

1. `perf_coeff.json`（现行 `linear14`）：14 个特征（元素计数 + ρ + ΔHf + 氧平衡产物）
   的线性回归，与移植前 `runtime/perf.py` 用的完全是同一套系数；
2. `perf_coeff_kamlet.json`（候选 `kamlet_noco`）：Q 仍用上面的回归，D/p 换成
   「**无 CO** 氧平衡 + 对数-对数 Kamlet 形式」——实测留出集把 D 的中位相对误差
   从 4.19% 降到 2.31%、p 从 5.35% 降到 4.62%。

两套都保留在推理核里，可由 `perf_variant` 切换并互相对照。

用法（开发侧 python，只需要 numpy；不再依赖 pandas）：
    python tools/fit_perf.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "模型" / "data" / "final_123679.csv"
CSV_ALT = ROOT / "model" / "data" / "final_123679.csv"
OUT_LEGACY = ROOT / "runtime" / "weights" / "perf_coeff.json"
OUT_FLAT = ROOT / "boom-data" / "perf_coeff.json"
OUT_KAMLET = ROOT / "boom-data" / "perf_coeff_kamlet.json"

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


def products_with_co(c, h, n, o):
    """含 CO 的氧平衡产物规则（与 runtime/perf.py 一致）。"""
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


def products_without_co(c, h, n, o):
    """**无 CO** 的氧平衡：H→H2O，余氧→CO2，再余氧→O2，N→N2。

    模型/README.md §5 记录的那组 Kamlet 系数（-0.5735 / 0.4666 / 1.3877）就是用这条
    规则拟合出来的；本工具重拟合得到 -0.563 / 0.520 / 1.451，与之吻合。
    """
    nH2O = h / 2.0
    rem = o - nH2O
    nCO2 = np.where(rem > 0, np.minimum(c, np.maximum(rem, 0) / 2.0), 0.0)
    nCO = np.zeros_like(c)
    nO2 = np.where(rem > 2 * c, (rem - 2 * c) / 2.0, 0.0)
    nN2 = n / 2.0
    ngas = nH2O + nCO2 + nCO + nO2 + nN2
    mass = nH2O * 18.015 + nCO2 * 44.01 + nCO * 28.01 + nO2 * 32.0 + nN2 * 28.014
    Mgas = np.where(ngas > 0, mass / np.maximum(ngas, 1e-9), 0.0)
    return nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas


def build():
    src = CSV if CSV.exists() else CSV_ALT
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    rec = [parse_formula(r["Formula"]) for r in rows]
    c = np.array([d.get("C", 0) for d in rec], dtype=float)
    h = np.array([d.get("H", 0) for d in rec], dtype=float)
    n = np.array([d.get("N", 0) for d in rec], dtype=float)
    o = np.array([d.get("O", 0) for d in rec], dtype=float)
    rho = np.array([float(r["density(g/cm3)"]) for r in rows])
    hof = np.array([float(r["solid_HOF(kcal/mol)"]) for r in rows])
    nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas = products_with_co(c, h, n, o)
    X = np.column_stack([np.ones_like(c), c, h, n, o, rho, hof,
                         nCO2, nH2O, nCO, nO2, nN2, ngas, Mgas])
    y = {k: np.array([float(r[col]) for r in rows]) for k, col in TARGETS.items()}
    mw = c * MW_ATOM["C"] + h * MW_ATOM["H"] + n * MW_ATOM["N"] + o * MW_ATOM["O"]
    _, _, _, _, _, ngas_nc, Mgas_nc = products_without_co(c, h, n, o)
    N_per_g = ngas_nc / np.maximum(mw, 1e-9)
    return X, y, len(rows), N_per_g, Mgas_nc, rho


def medrel(pred, truth):
    return float(np.median(np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-9)))


def fit_linear14(X, y, tr, val):
    out = {}
    for name in TARGETS:
        truth = y[name][val]
        best = None
        for mode in ("linear", "log"):
            if mode == "log" and np.any(y[name][tr] <= 0):
                continue
            t = y[name][tr] if mode == "linear" else np.log(y[name][tr])
            coef, *_ = np.linalg.lstsq(X[tr], t, rcond=None)
            pred = X[val] @ coef
            if mode == "log":
                pred = np.exp(pred)
            cand = {"mode": mode, "coef": [float(v) for v in coef],
                    "median_rel": medrel(pred, truth)}
            if best is None or cand["median_rel"] < best["median_rel"]:
                best = cand
        pred = X[val] @ np.array(best["coef"])
        if best["mode"] == "log":
            pred = np.exp(pred)
        r2 = float(1 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean()) ** 2))
        best["holdout"] = {"mae": float(np.mean(np.abs(pred - truth))),
                           "median_rel": best.pop("median_rel"),
                           "r2": r2, "n": int(len(val))}
        out[name] = best
        print(f"linear14 {name:2s} [{best['mode']:6s}]: 中位相对误差 "
              f"{best['holdout']['median_rel'] * 100:.3f}%  MAE={best['holdout']['mae']:.4f}  "
              f"R2={r2:.5f}")
    return out


def fit_kamlet(X, y, tr, val, N_per_g, Mgas, rho):
    """Q 沿用 linear14 的回归；D/p 用无 CO 规则的 Kamlet 对数-对数形式。"""
    q_coef = np.array(y["Q_coef"])
    qhat = X @ q_coef
    K = np.log(np.maximum(N_per_g, 1e-12) * np.sqrt(Mgas) * np.sqrt(np.maximum(qhat, 1.0)))

    out = {}
    for name, extra in (("D", np.log(1 + 1.30 * rho)), ("p", np.log(rho))):
        basis = np.column_stack([np.ones_like(K), K, extra])
        coef, *_ = np.linalg.lstsq(basis[tr], np.log(y[name][tr]), rcond=None)
        pred = np.exp(basis @ coef)
        truth = y[name][val]
        r2 = float(1 - np.sum((pred[val] - truth) ** 2) / np.sum((truth - truth.mean()) ** 2))
        out[name] = {
            "basis": ("bias, ln(N*M^0.5*Q^0.5), ln(1+1.30*rho)" if name == "D"
                      else "bias, ln(N*M^0.5*Q^0.5), ln(rho)"),
            "coef": [float(v) for v in coef],
            "holdout": {"mae": float(np.mean(np.abs(pred[val] - truth))),
                        "median_rel": medrel(pred[val], truth), "r2": r2, "n": int(len(val))},
        }
        print(f"kamlet   {name:2s} [log   ]: 中位相对误差 "
              f"{out[name]['holdout']['median_rel'] * 100:.3f}%  "
              f"MAE={out[name]['holdout']['mae']:.4f}  R2={r2:.5f}  系数={np.round(coef, 6)}")
    return out


def main():
    X, y, n_all, N_per_g, Mgas, rho = build()
    rng = np.random.default_rng(641)
    idx = rng.permutation(n_all)
    n_val = n_all // 5
    val, tr = idx[:n_val], idx[n_val:]
    print(f"数据集 {n_all} 条：train {len(tr)} / val {len(val)}（seed 641，与原实现一致）\n")

    lin = fit_linear14(X, y, tr, val)

    legacy = {
        "desc": "由 tools/fit_perf.py 在随包数据集上拟合的经验关系（非模型输出）",
        "source_csv": "模型/data/final_123679.csv",
        "features": FEATURES,
        "products_rule": "CHNO 氧平衡：H→H2O，余氧→CO2，再余氧→O2，N→N2",
        "targets": {k: {"column": TARGETS[k], **v} for k, v in lin.items()},
    }
    legacy["note"] = ("Q/D/p 只由 (分子式, 密度, 生成焓) 经验拟合，同分异构体差异无法体现；"
                      "仅作量级参考，不是模型输出。")

    y2 = dict(y)
    y2["Q_coef"] = lin["Q"]["coef"]
    kam = fit_kamlet(X, y2, tr, val, N_per_g, Mgas, rho)
    kamlet_doc = {
        "desc": "Q/D/p 候选实现：无 CO 氧平衡 + 对数-对数 Kamlet 形式（非模型输出）",
        "variant": "kamlet_noco",
        "source_csv": "模型/data/final_123679.csv",
        "products_rule": "无 CO 氧平衡：H→H2O，余氧→CO2，再余氧→O2，N→N2",
        "Q_from": "perf_coeff.json 的 Q（14 特征线性回归）",
        **kam,
        "note": ("D/p 用 Kamlet 形式 ln = a + b·ln(N·M^0.5·Q^0.5) + c·ln(1+1.30ρ)/c·ln(ρ)；"
                 "Q 仍取 linear14 的结果。两套结果在 UI 里可同时显示、互相对照。"),
    }

    for path in (OUT_LEGACY, OUT_FLAT, OUT_KAMLET):
        path.parent.mkdir(parents=True, exist_ok=True)
    OUT_LEGACY.write_text(json.dumps(legacy, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_FLAT.write_text(json.dumps(legacy, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_KAMLET.write_text(json.dumps(kamlet_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {OUT_LEGACY}\nwritten: {OUT_FLAT}\nwritten: {OUT_KAMLET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
