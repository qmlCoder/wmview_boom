"""门禁：`boom-core/src/basis.rs` 的数值常量必须与生成物 `runtime/basis_gen.py` 完全一致。

背景：`tools/gen_basis.py` 用 sympy 把 模型/utils/special.py 的基函数固化成
`runtime/basis_gen.py`（纯 numpy 数值代码）。移植到 Rust 时，`basis.rs` 里的表达式与
系数是**从那份生成物誊写**过来的——誊写会漂移，而漂移会让数值悄悄偏掉（不至于崩，
但会破坏对拍）。

本脚本只做一件事：把两个文件里所有「长浮点字面量」（>=6 位有效数字，即真正的基函数
系数）抽出来做集合比较，必须完全相同。这样 `gen_basis.py` 一改就能立刻发现 Rust 没跟上。

用法：
    python tools/check_basis_parity.py
    python tools/gen_basis.py && python tools/check_basis_parity.py   # 重新生成后再查
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_SRC = ROOT / "runtime" / "basis_gen.py"
RS_SRC = ROOT / "boom-core" / "src" / "basis.rs"

# 长浮点字面量：整数部分 + 小数点 + 至少 6 位小数（基函数系数都是这种形态）
LITERAL = re.compile(r"-?\d+\.\d{6,}(?:e-?\d+)?", re.IGNORECASE)


def literals(path: Path, start_marker: str | None = None) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if start_marker:
        idx = text.find(start_marker)
        if idx < 0:
            raise SystemExit(f"{path} 里找不到标记 {start_marker}")
        text = text[idx:]
    out = set()
    for m in LITERAL.finditer(text):
        raw = m.group(0)
        # 统一成浮点规范形式，避免写法差异（如末尾 0 的个数）
        out.add(repr(float(raw)))
    return out


def main() -> int:
    if not PY_SRC.exists() or not RS_SRC.exists():
        raise SystemExit(f"缺文件：{PY_SRC} 或 {RS_SRC}")

    # 只比对**函数体**里的系数：basis_gen.py 顶部的 BESSEL_ZEROS / BESSEL_NORM 两张表是
    # 原实现遗留的死数据（生成出的 sbf_radial/sph_sbf/sph_tbf 用的是内联字面量），
    # Rust 侧没有对应表，不应该因此判不一致。
    py = literals(PY_SRC, start_marker="def sbf_radial")
    rs = literals(RS_SRC)
    print(f"runtime/basis_gen.py（函数体）: {len(py)} 个长常量")
    print(f"boom-core/src/basis.rs:        {len(rs)} 个长常量")

    only_py = sorted(py - rs)
    only_rs = sorted(rs - py)
    if only_py:
        print(f"\n[不一致] 生成物里有、Rust 里没有（{len(only_py)} 个）:")
        for v in only_py:
            print("   ", v)
    if only_rs:
        print(f"\n[不一致] Rust 里有、生成物里没有（{len(only_rs)} 个）:")
        for v in only_rs:
            print("   ", v)

    if only_py or only_rs:
        print("\n判定：不通过（basis.rs 与 tools/gen_basis.py 的产物已经漂移，请同步）")
        return 1
    print("\n判定：通过（Rust 基组常量与生成物完全一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
