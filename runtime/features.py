"""分子 -> 模型输入（键图 + 49 维元素描述子）。

对应 模型/data/features.py 里的 MyDataset.__getitem__ / get()：
- 原实现从 SMILES 用 RDKit AddHs 拿键图、从 mendeleev 现算元素描述子；
- 这里键图由宿主（wmview 的化学键感知）或距离判据给出，描述子查固化表。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DT = np.float32

# 原实现用到的 8 个元素（ele_nums = [1] + list(range(3, 10))）
SYMBOLS = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F']
ATOMIC_NUMBER = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
                 'N': 7, 'O': 8, 'F': 9}

# 共价半径（Å），用于「距离判据」建键
COVALENT_RADIUS = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84,
                   'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57}


def load_table(weights_dir) -> np.ndarray:
    """读取固化的 49 维元素 one-hot 表（tools/freeze_element_table.py 生成）。"""
    path = Path(weights_dir) / 'element_table.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    table = np.asarray(data['table'], dtype=DT)
    if table.shape != (8, 49):
        raise ValueError(f'元素描述子表形状异常: {table.shape}（应为 (8, 49)）')
    return table


def element_rows(syms, table, index_mode='legacy'):
    """元素符号 -> 描述子表行号。

    legacy（默认）：row = Z - 1。原代码 `n = number - 1` 的 off-by-one 就这样，
    权重是在这套「错位」特征上训练的（见 模型/README.md §6.2 第 6 条），
    所以必须原样复现才能和原模型对齐：C 用 N 的行、N 用 O 的行、O 用 F 的行。
    corrected：按元素在 SYMBOLS 中的正确位置取行（仅供对照，输出与权重不匹配）。
    """
    rows = []
    for s in syms:
        if s not in ATOMIC_NUMBER:
            raise ValueError(f'不支持的原子类型: {s}')
        if index_mode == 'corrected':
            if s not in SYMBOLS:
                raise ValueError(f'元素 {s} 不在固化的 8 元素表内')
            row = SYMBOLS.index(s)
        else:
            row = ATOMIC_NUMBER[s] - 1
            if not 0 <= row < table.shape[0]:
                raise ValueError(f'元素 {s}（Z={ATOMIC_NUMBER[s]}）超出原实现的索引范围'
                                 f'（复现 n=Z-1，只支持 Z<=8 的 H/Li/Be/B/C/N/O）')
        rows.append(row)
    return rows


def build_x_geom(syms, table, index_mode='legacy') -> np.ndarray:
    rows = element_rows(syms, table, index_mode)
    return table[rows, :].astype(DT)


def hill_formula(syms) -> str:
    counts = {}
    for s in syms:
        counts[s] = counts.get(s, 0) + 1
    parts = []
    if 'C' in counts:
        parts.append(('C', counts.pop('C')))
    if 'H' in counts:
        parts.append(('H', counts.pop('H')))
    for s in sorted(counts):
        parts.append((s, counts[s]))
    return ''.join(f'{s}{n if n > 1 else ""}' for s, n in parts)


def formula_counts(syms) -> dict:
    counts = {}
    for s in syms:
        counts[s] = counts.get(s, 0) + 1
    return counts


def distance_bonds(syms, xyzs, scale=1.25):
    """距离判据建键：d < scale * (r_i + r_j) 视为成键（返回去重的无向键列表）。"""
    pos = np.asarray(xyzs, dtype=np.float64)
    n = pos.shape[0]
    bonds = []
    if n < 2:
        return bonds
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    for a in range(n):
        ra = COVALENT_RADIUS.get(syms[a])
        if ra is None:
            continue
        for b in range(a + 1, n):
            rb = COVALENT_RADIUS.get(syms[b])
            if rb is None:
                continue
            if d[a, b] < scale * (ra + rb):
                bonds.append([a, b])
    return bonds


def edge_index_from_bonds(bonds, n_atoms):
    """无向键列表 -> 双向 edge_index (j, i)（与原实现 RDKit 的 [i,j]/[j,i] 一致）。"""
    if bonds is None or len(bonds) == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64))
    b = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    keep = (b[:, 0] >= 0) & (b[:, 0] < n_atoms) & (b[:, 1] >= 0) & (b[:, 1] < n_atoms)
    b = b[keep]
    b = b[b[:, 0] != b[:, 1]]
    j = np.concatenate([b[:, 0], b[:, 1]])
    i = np.concatenate([b[:, 1], b[:, 0]])
    return j, i
