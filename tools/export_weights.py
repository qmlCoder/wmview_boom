"""把 PyTorch checkpoint（模型/model_save/*.pt）转成推理运行时可读的 npz。

运行时（runtime/）不带 torch，只带 numpy，所以权重必须先离线转换。
转换过程只做 `state_dict -> numpy float32 数组`，不改数值。

用法（需要 torch）：
    python tools/export_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
# 原项目快照目录是中文名 `模型/`；若某天被重命名成 ASCII 的 `model/` 也能用。
# 注意判断的是**父目录**是否存在，不是 `model_save/`——新克隆的仓库里
# `模型/` 有、`模型/model_save/` 没有（该目录只装被 gitignore 的 .pt，git 不保留空目录）。
MODEL_BASE = ROOT / "模型"
MODEL_BASE_ALT = ROOT / "model"
OUT_DIR = ROOT / "runtime" / "weights"

TARGETS = {
    "density": {"file": "best_density.pt", "label": "密度 ρ (g/cm³)", "unit": "g/cm^3"},
    "eof": {"file": "best_eof.pt", "label": "固体生成焓 ΔHf (kcal/mol)", "unit": "kcal/mol"},
}


def model_dir() -> Path:
    """checkpoint 目录：优先用中文名 `模型/model_save/`，没有 `模型/` 时退回 `model/model_save/`。

    新克隆的仓库里 `模型/model_save/` **不存在**（该目录只装被 gitignore 的 .pt），
    所以这里按需新建，避免下一步 open() 抛 FileNotFoundError。
    """
    base = MODEL_BASE if MODEL_BASE.exists() else MODEL_BASE_ALT
    d = base / "model_save"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_ckpt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - 老版本 torch 或含自定义对象时回退
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    src = model_dir()
    missing = [info["file"] for info in TARGETS.values() if not (src / info["file"]).is_file()]
    if missing:
        raise SystemExit(
            "缺 checkpoint：" + "、".join(missing) + "\n"
            f"请把 best_density.pt / best_eof.pt 放进 {src} 再跑本脚本。\n"
            "（仓库按同级插件的约定不提交大二进制：*.pt / *.npz / boom-data/*.bin 都在 .gitignore 里，"
            "新克隆下来没有这两个文件。）"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metas = {}
    key_sets = {}
    for name, info in TARGETS.items():
        path = src / info["file"]
        ckpt = load_ckpt(path)
        state = ckpt["model"]
        arrays = {}
        for k, v in state.items():
            arrays[k] = np.asarray(v.detach().cpu().numpy(), dtype=np.float32)
        out = OUT_DIR / f"{name}.npz"
        np.savez(out, **arrays)
        n_param = int(sum(a.size for a in arrays.values()))
        key_sets[name] = set(arrays)
        metas[name] = {
            "source": info["file"],
            "label": info["label"],
            "unit": info["unit"],
            "epoch": int(ckpt["epoch"]),
            "val_mae": float(ckpt["val_mae"]),
            "n_param": n_param,
            "npz": f"{name}.npz",
        }
        print(f"{name}: {len(arrays)} tensors, {n_param} params -> {out.name} "
              f"({out.stat().st_size / 1e6:.1f} MB)  epoch={metas[name]['epoch']} "
              f"val_mae={metas[name]['val_mae']:.4f}")

    same = key_sets["density"] == key_sets["eof"]
    print("两个 checkpoint 的 key 完全一致:", same)
    if not same:
        raise SystemExit("checkpoint 结构不一致，请检查")

    # 模型结构摘要（供 runtime 自检 / 文档使用）
    meta = {
        "desc": "GRMP 权重（由 tools/export_weights.py 从 .pt 转换，数值未改动）",
        "arch": {
            "hidden_channels": 128,
            "pass_layers": 3,
            "node_embedding_in": int(np.load(OUT_DIR / "density.npz")["node_embedding.weight"].shape[1]),
            "adaptive_type": "gate",
            "cutoff": 5.0,
            "g_cutoff": 7.0,
            "num_spherical": 3,
            "num_radial": 6,
            "envelope_exponent": 5,
        },
        "targets": metas,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written:", OUT_DIR / "meta.json")


if __name__ == "__main__":
    main()
