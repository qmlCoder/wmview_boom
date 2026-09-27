"""把原模型 data/features.py 里「推理时现算」的 49 维元素描述子固化成常量表。

原实现的 `atom()` 每跑一次都会用 mendeleev 重新取 8 个元素（H, Li, Be, B, C, N, O, F）的
7 个物理量，再用 sklearn 的 OneHotEncoder 现场 one-hot。mendeleev 数据一更新列序就可能变，
权重会**静默失效**（见 模型/README.md §6.2 第 7 条）。这里把结果固化成 runtime/weights/
element_table.json，推理时只查表。

注意：one-hot 的列只取决于每个物理量在 8 个元素上的**排序**，与数值大小无关，
所以只要排序一致，固化表与原实现等价（工具会顺带打印排序结果便于核对）。

用法（需要 mendeleev）：
    python tools/freeze_element_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runtime" / "weights" / "element_table.json"

# 原代码：ele_nums = [1] + list(range(3, 10))
ELE_NUMS = [1] + list(range(3, 10))
SYMBOLS = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F']


def raw_feature_lists():
    """逐行对应 模型/data/features.py 的 atom()。"""
    from mendeleev import element

    feature_lists = {
        'number': [],
        'ionenergies': [],
        'atomic_radius': [],
        'period_list': [],
        'nvalence': [],
        'atomic_volume': [],
        'electron_affinity': [],
    }
    for i in ELE_NUMS:
        ele = element(i)
        feature_lists['number'].append(i)
        feature_lists['ionenergies'].append(ele.ionenergies[1])
        feature_lists['atomic_radius'].append(float(ele.atomic_radius))
        feature_lists['period_list'].append(ele.period)
        feature_lists['nvalence'].append(ele.nvalence())
        feature_lists['atomic_volume'].append(float(ele.atomic_volume))
        feature_lists['electron_affinity'].append(float(ele.electron_affinity))
    return feature_lists


def one_hot_columns(values):
    """等价于 sklearn OneHotEncoder().fit(...).transform(...).toarray()（数值列按升序排）。"""
    uniq = sorted(set(values))
    index = {v: i for i, v in enumerate(uniq)}
    out = np.zeros((len(values), len(uniq)), dtype=np.float64)
    for r, v in enumerate(values):
        out[r, index[v]] = 1.0
    return out, uniq


def main():
    feature_lists = raw_feature_lists()
    cols = []
    meta = {}
    for name, values in feature_lists.items():
        block, uniq = one_hot_columns(values)
        cols.append(block)
        meta[name] = {
            'values': values,
            'sorted_unique': uniq,
            'width': block.shape[1],
        }
        order = sorted(range(len(values)), key=lambda i: values[i])
        print(f'{name:20s} width={block.shape[1]}  升序: '
              + ', '.join(f'{SYMBOLS[i]}={values[i]}' for i in order))
    table = np.concatenate(cols, axis=1)
    assert table.shape == (8, 49), table.shape

    data = {
        'desc': 'GRMP 输入 x_geom 的 49 维元素 one-hot 描述子（固化自 features.py 的 atom()）',
        'source': 'mendeleev + sklearn OneHotEncoder（见 tools/freeze_element_table.py）',
        'symbols': SYMBOLS,
        'atomic_numbers': ELE_NUMS,
        'properties': list(feature_lists.keys()),
        'property_meta': meta,
        # 原代码 n = number - 1，是已知 off-by-one（见 模型/README.md §6.2 第 6 条），
        # 权重就是在这套「错位」特征上训练出来的，默认必须原样复现才能对齐。
        'legacy_index': 'row = Z - 1',
        'corrected_index': 'row = position of element in symbols',
        'table': table.tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwritten: {OUT}  table shape = {table.shape}')
    print('total dim =', table.shape[1])


if __name__ == '__main__':
    main()
