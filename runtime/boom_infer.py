"""wmview.boom 推理入口：从 stdin 读分子，跑 GRMP 模型，把结果写成 JSON。

调用方式（宿主 wmapi_cores.run_exe 会把它当外部程序执行，input 写进 stdin）：
    pythonw.exe <插件目录>/runtime/boom_infer.py --out <插件目录>/runtime/out/result.json

stdin（JSON）:
    {
      "name": "gdb_1.xyz",             # 可选，仅用于显示
      "syms": ["C","H","N","O",...],
      "xyzs": [[x,y,z], ...],
      "bonds": [[i,j], ...] | null,    # 宿主给的化学键，可为空
      "bond_mode": "auto",             # auto | host | distance（auto=有宿主键就用宿主键）
      "targets": ["density","eof"],    # 要跑的模型
      "derived": true,                 # 是否估算 Q / D / p（经验关系，非模型输出）
      "index_mode": "legacy"           # legacy=复现原实现 n=Z-1 的错位；corrected=按元素正确取行
    }

stdout：人读的报告（宿主按 GBK 解码后进日志面板）
result.json：机器读的结果（UTF-8）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 宿主 run_exe 按 GBK 解码 stdout；中文 Windows 下 python 默认也是 GBK，
# 但个别字符（如 ³）编不出来，这里统一用 replace 兜底，避免整个报告打印失败
if sys.platform == 'win32':
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

import numpy as np  # noqa: E402

import features as F  # noqa: E402
import grmp_numpy as G  # noqa: E402
import perf as P  # noqa: E402

# 训练集覆盖域：下面是 模型/data/final_123679.csv（123679 条）的实测统计，
# 含「每个元素各自的出现范围」和「哪些元素是所有训练分子都有的」。
# 注意 C/N/O：训练集里 100% 的分子都含 C、N、O 各至少 1 个（没有 N-free / O-free 的样本），
# H 有 8 条样本为 0；重原子数只有 8/9/10。
DOMAIN = {
    'C': (1, 7), 'H': (0, 8), 'N': (1, 5), 'O': (1, 4),
    'heavy': (8, 10), 'atoms': (8, 17), 'rho': (1.2167, 1.9570),
}
ALWAYS_PRESENT = ('C', 'N', 'O')   # 训练集里每个分子都含这些元素
ELEMENTS = ('C', 'H', 'N', 'O')


def write_result(out_path: Path, payload: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')


def domain_warnings(syms, counts, rho=None):
    warns = []
    other = [s for s in counts if s not in ('C', 'H', 'N', 'O')]
    if other:
        warns.append(f'含非 CHNO 元素 {sorted(set(other))}：远超训练集覆盖域，结果不可信')
    # 1) 训练集里必有的元素缺失（N/O 缺失 = 完全没见过的分子类型）
    absent = [el for el in ALWAYS_PRESENT if counts.get(el, 0) == 0]
    if absent:
        warns.append(f"本分子不含 {'/'.join(absent)}，而训练集里所有分子都含 C/N/O"
                     f"（没有 N-free / O-free 的样本）→ 属外推")
    # 2) 各元素原子数上限
    over = [f'{el}={counts[el]}（训练集上限 {DOMAIN[el][1]}）'
            for el in ELEMENTS if counts.get(el, 0) > DOMAIN[el][1]]
    if over:
        warns.append('原子数超出训练集范围：' + '，'.join(over))
    # 3) 重原子数（模型真正“见过”的分子规模）
    n_heavy = sum(counts.get(el, 0) for el in ELEMENTS if el != 'H')
    lo, hi = DOMAIN['heavy']
    if not (lo <= n_heavy <= hi):
        warns.append(f'重原子数 {n_heavy} 超出训练集范围 [{lo},{hi}]'
                     f'（训练集只有 8~10 个重原子的小分子）')
    n_atoms = len(syms)
    lo, hi = DOMAIN['atoms']
    if not (lo <= n_atoms <= hi):
        warns.append(f'原子总数 {n_atoms} 超出训练集范围 [{lo},{hi}]（训练集 8~17 个原子）')
    if rho is not None:
        lo, hi = DOMAIN['rho']
        if not (lo <= rho <= hi):
            warns.append(f'预测密度 {rho:.3f} 超出训练集实测范围 [{lo},{hi}]，属外推、不要采信')
    return warns


def run(payload: dict, weights_dir: Path, index_mode: str):
    syms = payload['syms']
    xyzs = np.asarray(payload['xyzs'], dtype=np.float64)
    if xyzs.ndim != 2 or xyzs.shape[1] != 3 or xyzs.shape[0] != len(syms):
        raise ValueError('syms / xyzs 形状不匹配')
    n_atoms = len(syms)

    bond_mode = payload.get('bond_mode', 'auto')
    host_bonds = payload.get('bonds') or None
    src = 'host'
    if bond_mode == 'distance' or (bond_mode == 'auto' and not host_bonds):
        bonds = F.distance_bonds(syms, xyzs)
        src = 'distance'
    else:
        bonds = host_bonds
    j, i = F.edge_index_from_bonds(bonds, n_atoms)

    table = F.load_table(weights_dir)
    x_geom = F.build_x_geom(syms, table, index_mode)

    counts = F.formula_counts(syms)
    target_files = {}
    for name in payload.get('targets') or ['density', 'eof']:
        path = weights_dir / f'{name}.npz'
        if not path.exists():
            raise ValueError(f'缺少权重文件 {path}')
        target_files[name] = path

    results = {}
    timings = {}
    for name, path in target_files.items():
        t0 = time.perf_counter()
        results[name] = G.predict(str(path), x_geom, xyzs, (j, i))
        timings[name] = (time.perf_counter() - t0) * 1000.0

    meta_path = weights_dir / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}

    derived = None
    if payload.get('derived', True) and 'density' in results and 'eof' in results:
        coeff = P.load_coeff(weights_dir)
        derived = P.estimate(counts, results['density'], results['eof'], coeff)

    warnings = domain_warnings(syms, counts, results.get('density'))
    if n_atoms and 'H' not in counts:
        warnings.append('分子中没有显式氢原子：原模型是在 AddHs 后的含氢结构上训练的，结果偏差会很大')

    return {
        'ok': True,
        'token': payload.get('token'),
        'name': payload.get('name', ''),
        'formula': F.hill_formula(syms),
        'n_atoms': n_atoms,
        'n_heavy': int(sum(v for k, v in counts.items() if k != 'H')),
        'counts': counts,
        'bond_mode': src,
        'n_bonds': int(len(bonds) if bonds is not None else 0),
        'index_mode': index_mode,
        'results': results,
        'derived': derived,
        'warnings': warnings,
        'timing_ms': timings,
        'meta': {k: {'epoch': v.get('epoch'), 'val_mae': v.get('val_mae'), 'label': v.get('label')}
                 for k, v in (meta.get('targets') or {}).items()},
    }


def report(res: dict) -> str:
    lines = []
    lines.append(f"分子 {res['name']}  式 {res['formula']}  "
                 f"原子 {res['n_atoms']}（重原子 {res['n_heavy']}）  键 {res['n_bonds']}（{res['bond_mode']}）")
    if 'density' in res['results']:
        m = res['meta'].get('density', {})
        lines.append(f"  密度 ρ      = {res['results']['density']:.4f} g/cm^3"
                     f"   （模型 best_density，epoch {m.get('epoch')}，val MAE {m.get('val_mae')}）")
    if 'eof' in res['results']:
        m = res['meta'].get('eof', {})
        lines.append(f"  生成焓 ΔHf  = {res['results']['eof']:.3f} kcal/mol"
                     f"   （模型 best_eof，epoch {m.get('epoch')}，val MAE {m.get('val_mae')}）")
    if res.get('derived'):
        d = res['derived']
        lines.append(f"  经验估算（非模型输出）：Q = {d['Q']:.1f} cal/g，"
                     f"D = {d['D']:.3f} km/s，p = {d['p']:.2f} GPa")
    for w in res.get('warnings', []):
        lines.append(f"  [警告] {w}")
    t = res.get('timing_ms') or {}
    if t:
        lines.append('  耗时 ' + '，'.join(f'{k} {v:.0f} ms' for k, v in t.items()))
    return '\n'.join(lines)


# 环境自检用的固定分子（硝基甲烷 CH3NO2 的近似几何，单位 Å）
CHECK_MOL = {
    'name': 'selftest(CH3NO2)',
    'syms': ['C', 'N', 'O', 'O', 'H', 'H', 'H'],
    'xyzs': [[0.0, 0.0, 0.0], [1.47, 0.0, 0.0], [2.08, 1.06, 0.0], [1.90, -1.10, 0.0],
             [-0.55, 0.87, 0.0], [-0.55, -0.48, 0.82], [-0.55, -0.48, -0.82]],
    'bond_mode': 'distance',
    'targets': ['density', 'eof'],
    'derived': True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(HERE / 'weights'))
    ap.add_argument('--out', default=str(HERE / 'out' / 'result.json'))
    ap.add_argument('--check', action='store_true', help='只做环境自检')
    args = ap.parse_args()
    weights_dir = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()

    try:
        if args.check:
            print(f'python {sys.version.split()[0]} / numpy {np.__version__}')
            print(f'weights dir: {weights_dir}')
            for name in ('density.npz', 'eof.npz', 'element_table.json', 'meta.json', 'perf_coeff.json'):
                p = weights_dir / name
                print(f'  {"OK " if p.exists() else "缺失"} {name}')
            res = run(CHECK_MOL, weights_dir, 'legacy')
            print(report(res))
            write_result(out_path, res)
            print('自检通过（以上数值仅是「能算出来」，不作精度说明）')
            return 0

        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else CHECK_MOL
        index_mode = payload.get('index_mode', 'legacy')
        res = run(payload, weights_dir, index_mode)
        print(report(res))
        write_result(out_path, res)
        return 0
    except Exception as exc:  # noqa: BLE001 - 任何异常都要落盘给前端看
        detail = traceback.format_exc()
        print('推理失败: ' + str(exc))
        print(detail)
        write_result(out_path, {'ok': False, 'error': str(exc), 'traceback': detail})
        return 1


if __name__ == '__main__':
    sys.exit(main())
