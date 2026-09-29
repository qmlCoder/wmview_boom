"""Rust 推理核 vs numpy 参照实现：逐分子交叉验证（**开发侧工具，不进包**）。

做法：拿同一批分子（默认 `tools/golden.json` 里那 8 个，键图直接给 RDKit 的键）
分别喂给两条链路，比较**整条管线**的每个字段，而不只是模型输出：

- numpy 链路：`runtime/boom_infer.py`（迁移期的参照实现，随包的运行时已经换成 Rust）
- Rust  链路：`boom-core/target/release/boom-core.exe`（stdin JSON -> stdout JSON）

比较项：density / eof、Q / D / p（linear14 与 kamlet 两套）、Hill 式、键数、
bond_mode、index_mode、覆盖域警告文本、n_atoms / n_heavy。

正门禁是 `boom-core --golden`（与原 PyTorch 参考输出对拍，阈值 1e-4）；
本脚本是更细的交叉核对，用于改数值实现时定位问题。

用法（需要装了 numpy 的 python，插件目录下已不再自带运行时）：
    python tools/compare_impls.py [分子数]        # 默认全部
    python tools/compare_impls.py --variant kamlet
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)
RUST = ROOT / "boom-core" / "target" / "release" / "boom-core.exe"
GOLDEN = ROOT / "tools" / "golden.json"
NUMPY_OUT = ROOT / "runtime" / "out" / "parity_numpy.json"
TOL = 1e-4


def run_numpy(payload: dict) -> dict:
    # 注意编码：python 侧 stdout 是 GBK（宿主也按 GBK 解码），Rust 侧是 UTF-8
    subprocess.run(
        [str(PY), str(ROOT / "runtime" / "boom_infer.py"), "--out", str(NUMPY_OUT)],
        input=json.dumps(payload), text=True, capture_output=True, check=False,
        encoding="gbk", errors="replace",
    )
    return json.loads(NUMPY_OUT.read_text(encoding="utf-8"))


def run_rust(payload: dict) -> dict:
    p = subprocess.run(
        [str(RUST), "--data", str(ROOT / "boom-data")],
        input=json.dumps(payload), text=True, capture_output=True, check=False,
        encoding="utf-8", errors="replace",
    )
    if p.returncode != 0:
        raise RuntimeError(f"boom-core 退出码 {p.returncode}：{p.stderr[-400:]}")
    return json.loads(p.stdout)


def rel(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-6)


def main() -> int:
    args = [a for a in sys.argv[1:]]
    variant = "linear14"
    if "--variant" in args:
        i = args.index("--variant")
        variant = args[i + 1]
        del args[i:i + 2]
    limit = int(args[0]) if args else None

    if not RUST.exists():
        raise SystemExit(f"找不到 {RUST}（先 cd boom-core && cargo build --release）")
    if not PY.exists():
        raise SystemExit(f"找不到 python: {PY}")
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit("当前 python 没有 numpy，numpy 参照链路跑不起来（用装了 numpy 的解释器）")

    mols = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if limit:
        mols = mols[:limit]
    print(f"对照分子 {len(mols)} 个，perf 主口径 = {variant}\n")

    worst = {"density": 0.0, "eof": 0.0, "Q": 0.0, "D": 0.0, "p": 0.0}
    worst_at = {k: "" for k in worst}
    problems: list[str] = []
    for m in mols:
        payload = {
            "name": m["name"],
            "syms": m["syms"],
            "xyzs": m["xyzs"],
            "bonds": m["bonds"],
            "bond_mode": "host",
            "index_mode": "legacy",
            "targets": ["density", "eof"],
            "derived": True,
            "perf_variant": variant,
        }
        np_res = run_numpy(payload)
        rs_res = run_rust(payload)

        line = [f"{m['name']:>16}"]
        for key in ("density", "eof"):
            a = np_res["results"].get(key)
            b = rs_res["results"].get(key)
            r = rel(a, b)
            if r > worst[key]:
                worst[key], worst_at[key] = r, m["name"]
            line.append(f"{key}: numpy={a:.6f} rust={b:.6f} 相对={r:.2e}")
        for key in ("Q", "D", "p"):
            a = (np_res.get("derived") or {}).get(key)
            b = (rs_res.get("derived") or {}).get(key)
            r = rel(a, b)
            if r > worst[key]:
                worst[key], worst_at[key] = r, m["name"]
            line.append(f"{key}: numpy={a:.4f} rust={b:.4f}")
        print("  ".join(line))

        # 非数值字段必须**完全一致**
        for key in ("formula", "n_atoms", "n_heavy", "bond_mode", "n_bonds"):
            if np_res.get(key) != rs_res.get(key):
                problems.append(f"{m['name']} {key}: numpy={np_res.get(key)!r} rust={rs_res.get(key)!r}")
        if np_res.get("warnings") != rs_res.get("warnings"):
            problems.append(f"{m['name']} warnings 不一致：\n    numpy={np_res.get('warnings')}\n    rust ={rs_res.get('warnings')}")
        if rs_res.get("perf_variant") != variant:
            problems.append(f"{m['name']} perf_variant 回显不符：{rs_res.get('perf_variant')}")
        # kamlet 那一套也一起比（Rust 会把「另一套」放在 derived_alt）
        alt = rs_res.get("derived_alt") or {}
        if variant == "linear14" and alt:
            print(f"{'':>16}  另一套(kamlet): Q={alt['Q']:.1f} D={alt['D']:.3f} p={alt['p']:.2f}")

    print("\n=== 数值最大相对偏差（numpy 为基准） ===")
    for k, v in worst.items():
        print(f"  {k:8s} {v:.3e}   最大处：{worst_at[k]}")

    used = {k: v for k, v in worst.items() if k in ("density", "eof")}
    ok = all(v < TOL for v in used.values()) and not problems
    print("\n=== 结构/文案一致性 ===")
    if problems:
        for p in problems:
            print("  [不一致] " + p)
    else:
        print("  formula / n_atoms / n_heavy / bond_mode / n_bonds / warnings 全部一致")
    print("\n判定：" + ("通过" if ok else "不通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
