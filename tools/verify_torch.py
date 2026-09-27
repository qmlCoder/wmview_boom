"""数值对齐验证：原 PyTorch 模型 vs runtime 的 numpy 实现。

做法：
1. 给 torch_cluster / torch_scatter / torch_sparse / torch_geometric 打最小替身；
2. **直接 import 原模型代码**（模型/model/mix.py 等，一行不改）；
3. 按 checkpoint 里的真实结构补上最后一层 MLP 的宽度，load_state_dict(strict=True)，
   这一步同时证明「我们对模型结构的理解」是对的；
4. 拿 data/final_123679.csv 里的分子（RDKit 建氢 + ETKDG 几何 + 同一个固化描述子表）
   分别跑 torch 和 numpy，比对两个模型（density / eof）的输出。

用法（需要 torch / rdkit）：
    python tools/verify_torch.py [样本数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'runtime'))

import _torch_stubs  # noqa: E402

_torch_stubs.install()

sys.path.insert(0, str(ROOT / '模型'))

from model.mix import GRMP  # noqa: E402  （原始模型代码）

import boom_infer as BI  # noqa: E402
import features as F  # noqa: E402
import grmp_numpy as G  # noqa: E402

WEIGHTS = ROOT / 'runtime' / 'weights'


def build_mlp_from_state(state):
    idx = sorted(int(k.split('.')[1]) for k in state
                 if k.startswith('MLP.') and k.endswith('.weight'))
    layers = []
    for n, i in enumerate(idx):
        w = state[f'MLP.{i}.weight']
        b = state.get(f'MLP.{i}.bias')
        layers.append(torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None))
        if n != len(idx) - 1:
            layers.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*layers)


def load_reference(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = ckpt['model']
    model = GRMP(hidden_channels=128, g_cutoff=7.0, cutoff=5.0, num_spherical=3, num_radial=6,
                 envelope_exponent=5, basis_emb_size_dist=128, basis_emb_size_angle=256,
                 basis_emb_size_torsion=512, emb_channels=256, pass_layers=3,
                 adaptive_type='gate')
    # 源码里的 MLP 宽度和 checkpoint 不一致（见 模型/README.md 的更正说明），
    # 这里按 checkpoint 的实际宽度重建，strict=True 加载成功即代表结构完全对上
    model.MLP = build_mlp_from_state(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def torch_forward(model, x_geom, pos, edge_index, batch_size):
    data = type('Data', (), {})()
    data.x_geom = torch.tensor(np.asarray(x_geom), dtype=torch.float32)
    data.pos = torch.tensor(np.asarray(pos), dtype=torch.float32)
    data.edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long)
    data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long)
    with torch.no_grad():
        return float(model(data)[0, 0])


def sample_molecules(n, seed=641):
    """从随包数据集取分子，用 RDKit 加氢 + ETKDG 生成几何，键图直接来自 RDKit（= 原实现）。"""
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    csv = ROOT / '模型' / 'data' / 'final_123679.csv'
    df = pd.read_csv(csv)
    # 取一些不同分子式的样本，保证结构多样性（含环、含硝基等）
    df = df.drop_duplicates(subset=['Formula']).sample(n=n, random_state=seed)
    out = []
    for _, row in df.iterrows():
        mol = Chem.AddHs(Chem.MolFromSmiles(row['smiles']))
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        if AllChem.EmbedMolecule(mol, params) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        syms = [a.GetSymbol() for a in mol.GetAtoms()]
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        out.append({'name': f'gdb_{row["number"]}.xyz', 'syms': syms, 'xyzs': pos.tolist(),
                    'bonds': bonds, 'bond_mode': 'host'})
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    table = F.load_table(WEIGHTS)
    refs = {name: load_reference(ROOT / '模型' / 'model_save' / f'best_{name}.pt')
            for name in ('density', 'eof')}
    print('两个 checkpoint 都用 strict=True 加载成功（结构理解正确）')

    mols = sample_molecules(n)
    print(f'测试分子 {len(mols)} 个')
    worst = {'density': 0.0, 'eof': 0.0}
    for mol in mols:
        syms = mol['syms']
        pos = np.asarray(mol['xyzs'], dtype=np.float64)
        j, i = F.edge_index_from_bonds(mol['bonds'], len(syms))
        x_geom = F.build_x_geom(syms, table, 'legacy')
        line = [f"{mol['name']:>14s} {F.hill_formula(syms):>10s} N={len(syms):2d} E={j.shape[0]:2d}"]
        for name in ('density', 'eof'):
            model, _ = refs[name]
            t_val = torch_forward(model, x_geom, pos, (j, i), 1)
            n_val = G.predict(str(WEIGHTS / f'{name}.npz'), x_geom, pos, (j, i))
            diff = abs(t_val - n_val)
            rel = diff / max(abs(t_val), 1e-6)
            worst[name] = max(worst[name], rel)
            line.append(f'{name}: torch={t_val:12.6f} numpy={n_val:12.6f} 绝对差={diff:.2e} 相对={rel:.2e}')
        print('\n  '.join(line))

    print('\n最大相对偏差:', {k: f'{v:.2e}' for k, v in worst.items()})
    ok = all(v < 1e-4 for v in worst.values())
    print('结论:', '通过（numpy 实现与原 PyTorch 模型在 float32 精度内一致）' if ok else '不通过')

    # 顺手把这次验证的分子与结果写成 golden 回归数据
    golden = []
    for mol in mols:
        x_geom = F.build_x_geom(mol['syms'], table, 'legacy')
        j, i = F.edge_index_from_bonds(mol['bonds'], len(mol['syms']))
        entry = {'name': mol['name'], 'syms': mol['syms'], 'xyzs': mol['xyzs'],
                 'bonds': mol['bonds']}
        for name in ('density', 'eof'):
            entry[name] = torch_forward(refs[name][0], x_geom, np.asarray(mol['xyzs']), (j, i), 1)
        golden.append(entry)
    out_path = HERE / 'golden.json'
    out_path.write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding='utf-8')
    print('golden 样本已写入:', out_path)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
