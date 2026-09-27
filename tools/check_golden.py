"""不依赖 torch 的回归自检：用 tools/golden.json 里记录的 PyTorch 参考值验证 numpy 运行时。

golden.json 由 tools/verify_torch.py 生成（含分子结构、键图、RDKit 几何与 torch 输出）。

用法：
    python tools/check_golden.py        # 可用内置 python（只依赖 numpy）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'runtime'))

import numpy as np  # noqa: E402

import features as F  # noqa: E402
import grmp_numpy as G  # noqa: E402

WEIGHTS = ROOT / 'runtime' / 'weights'
# verify_torch.py 的判定阈值：float32 前向的正常数值噪声
TOL = 1e-4


def main():
    golden = json.loads((HERE / 'golden.json').read_text(encoding='utf-8'))
    table = F.load_table(WEIGHTS)
    worst = 0.0
    for mol in golden:
        syms = mol['syms']
        pos = np.asarray(mol['xyzs'], dtype=np.float64)
        j, i = F.edge_index_from_bonds(mol['bonds'], len(syms))
        x_geom = F.build_x_geom(syms, table, 'legacy')
        for name in ('density', 'eof'):
            ref = float(mol[name])
            got = G.predict(str(WEIGHTS / f'{name}.npz'), x_geom, pos, (j, i))
            rel = abs(got - ref) / max(abs(ref), 1e-6)
            worst = max(worst, rel)
            flag = 'ok ' if rel < TOL else 'FAIL'
            print(f'{flag} {mol["name"]:>16s} {name:8s} torch={ref:12.6f} numpy={got:12.6f} '
                  f'相对差={rel:.2e}')
    print(f'\n最大相对偏差 {worst:.2e}（阈值 {TOL:.0e}）')
    if worst >= TOL:
        raise SystemExit('回归自检不通过')
    print('回归自检通过')


if __name__ == '__main__':
    main()
