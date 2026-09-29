"""把 runtime/weights/*.npz 转成 Rust/WASM 推理核用的**扁平权重**。

为什么不用 npz：npz 是 zip 容器，wasm 里解压不划算（与 wmvie.flavor 的
flavor-core/src/weights.rs 同一理由）。扁平格式 = 一个索引 json + 一个 f32 小端
bin，运行时一次 fetch 就能直接当成 f32 切片用，省掉 zlib 依赖。

产物（boom-data/，随包发布，由 Vite 以 ?url 资源引入）：
    density.json / density.bin     密度模型
    eof.json     / eof.bin         生成焓模型
    element_table.json             49 维元素描述子表（从 runtime/weights 拷过来）
    *_f16.bin                      fp16 对照（只作体积/精度实验，默认不用）

索引 json 结构：
    {
      "target": "density",
      "label": "密度 ρ (g/cm³)", "unit": "g/cm^3",
      "epoch": 44, "val_mae": 0.0202435, "n_param": 3805320,
      "arch": {...},                        # 与 runtime/weights/meta.json 一致
      "dtype": "f32",
      "tensors": {"node_embedding.weight": {"shape": [128, 49], "offset": 0}, ...}
    }
张量按名字典序排列，offset 以**元素个数**计（f32 时字节偏移 = offset*4）。

用法（开发侧 python，只需要 numpy；插件目录下已不再自带运行时）：
    python tools/export_flat_weights.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runtime" / "weights"
OUT = ROOT / "boom-data"

TARGETS = ("density", "eof")


def export_one(name: str, meta: dict) -> dict:
    data = np.load(SRC / f"{name}.npz")
    keys = sorted(data.files)
    tensors: dict = {}
    offset = 0
    chunks_f32 = []
    chunks_f16 = []
    for k in keys:
        a = np.ascontiguousarray(data[k], dtype="<f4")
        tensors[k] = {"shape": list(a.shape), "offset": offset}
        flat = a.reshape(-1)
        chunks_f32.append(flat)
        chunks_f16.append(np.ascontiguousarray(data[k], dtype="<f2").reshape(-1))
        offset += flat.size
    blob = np.concatenate(chunks_f32).tobytes()
    (OUT / f"{name}.bin").write_bytes(blob)
    (OUT / f"{name}_f16.bin").write_bytes(np.concatenate(chunks_f16).tobytes())

    tinfo = (meta.get("targets") or {}).get(name, {})
    index = {
        "target": name,
        "label": tinfo.get("label", name),
        "unit": tinfo.get("unit", ""),
        "epoch": tinfo.get("epoch"),
        "val_mae": tinfo.get("val_mae"),
        "n_param": int(offset),
        "arch": meta.get("arch", {}),
        "dtype": "f32",
        "tensors": tensors,
    }
    (OUT / f"{name}.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    size = len(blob) / 1e6
    print(f"{name:8s}: {len(keys):3d} 张量 / {offset} 参数 -> {name}.bin {size:.2f} MB "
          f"(fp16 {size / 2:.2f} MB)")
    return index


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))

    indexes = {name: export_one(name, meta) for name in TARGETS}

    # 两个 checkpoint 的张量名与形状必须完全一致（否则说明导错了）
    shapes = [{k: tuple(v["shape"]) for k, v in ix["tensors"].items()} for ix in indexes.values()]
    if shapes[0] != shapes[1]:
        raise SystemExit("density / eof 的张量名或形状不一致，导出有问题")
    print(f"两个模型的张量名与形状完全一致：{len(shapes[0])} 个张量")

    shutil.copyfile(SRC / "element_table.json", OUT / "element_table.json")
    print(f"已拷贝 element_table.json ({(OUT / 'element_table.json').stat().st_size / 1024:.1f} KB)")

    total = sum((OUT / f"{n}.bin").stat().st_size for n in TARGETS)
    print(f"扁平权重合计 {total / 1e6:.2f} MB（fp16 版本减半；是否启用需先过对拍）")
    print("输出目录:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
