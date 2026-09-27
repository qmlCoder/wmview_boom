"""图 / 几何特征（numpy 版），逐行对应 模型/utils/special.py 与 模型/layer/geom.py。

原实现依赖 torch_sparse.SparseTensor 做「按目标节点取边」的操作，这里用
等价的 numpy 索引实现（顺序语义一致：保持边的输入顺序）。
"""

from __future__ import annotations

import numpy as np

DT = np.float32


def remove_self_loops(j, i):
    """对应 torch_geometric.utils.remove_self_loops（只去自环，不去重）。"""
    keep = j != i
    return j[keep], i[keep]


def _edges_by_target(i, num_nodes):
    """by_target[t] = 「终点为 t」的边索引列表，保持边序。

    等价 torch_sparse 的 adj_t[t]（SparseTensor(row=col, col=row, value=arange) 的按行取值）。
    """
    order = np.argsort(i, kind='stable')
    counts = np.bincount(i, minlength=num_nodes)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return [order[starts[t]:starts[t] + counts[t]] for t in range(num_nodes)]


def _cat(arrays, dtype=np.int64):
    if len(arrays) == 0:
        return np.zeros((0,), dtype=dtype)
    return np.concatenate(arrays).astype(dtype, copy=False)


def geometry_indices(j, i, num_nodes):
    """对应 utils/special.py:get_geometry_indices。

    返回三体 (i-j-k) 与四体 (i-j-k-l) 的索引，以及「四体 -> 三体」的映射。
    """
    e = j.shape[0]
    by_target = _edges_by_target(i, num_nodes)

    # ---- i-j-k ----
    triple_counts = np.array([by_target[j[k]].shape[0] for k in range(e)], dtype=np.int64)
    idx_i = np.repeat(i, triple_counts)
    idx_j = np.repeat(j, triple_counts)
    idx_k = _cat([j[by_target[j[k]]] for k in range(e)])
    idx_kj = _cat([by_target[j[k]] for k in range(e)])

    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj = idx_kj[mask]
    idx_ji = np.repeat(np.arange(e, dtype=np.int64), triple_counts)[mask]

    # ---- i-j-k-l ----
    quad_counts = np.array([by_target[k].shape[0] for k in idx_k], dtype=np.int64)
    idx_i_t = np.repeat(idx_i, quad_counts)
    idx_j_t = np.repeat(idx_j, quad_counts)
    idx_k_t = np.repeat(idx_k, quad_counts)
    idx_l = _cat([j[by_target[k]] for k in idx_k])
    edge_kl = _cat([by_target[k] for k in idx_k])

    mask_q = (idx_l != idx_i_t) & (idx_l != idx_j_t)
    idx_i_t, idx_j_t, idx_k_t, idx_l = (
        idx_i_t[mask_q], idx_j_t[mask_q], idx_k_t[mask_q], idx_l[mask_q])
    idx_kj_t = np.repeat(idx_kj, quad_counts)[mask_q]
    idx_ji_t = np.repeat(idx_ji, quad_counts)[mask_q]
    idx_kl_t = edge_kl[mask_q]

    # 四体 -> 三体 的映射：等价 match.max(dim=1)[1]（全 False 时取 0）
    angle_keys = np.stack([idx_i, idx_j, idx_k], axis=1)
    quad_keys = np.stack([idx_i_t, idx_j_t, idx_k_t], axis=1)
    if quad_keys.shape[0] == 0:
        angle_idx_map = np.zeros((0,), dtype=np.int64)
    elif angle_keys.shape[0] == 0:
        angle_idx_map = np.zeros((quad_keys.shape[0],), dtype=np.int64)
    else:
        match = np.all(quad_keys[:, None, :] == angle_keys[None, :, :], axis=2)
        angle_idx_map = np.argmax(match, axis=1).astype(np.int64)

    return {
        'idx_i': idx_i, 'idx_j': idx_j, 'idx_k': idx_k,
        'idx_kj': idx_kj, 'idx_ji': idx_ji,
        'idx_i_t': idx_i_t, 'idx_j_t': idx_j_t, 'idx_k_t': idx_k_t, 'idx_l': idx_l,
        'idx_kj_t': idx_kj_t, 'idx_ji_t': idx_ji_t, 'idx_kl_t': idx_kl_t,
        'angle_idx_map': angle_idx_map,
    }


def compute_geometry_features(pos, j, i):
    """对应 utils/special.py:compute_geometry_features（输入是键图 edge_index=(j,i)）。"""
    pos = np.asarray(pos, dtype=np.float64)
    dist = np.linalg.norm(pos[i] - pos[j], axis=1)
    gi = geometry_indices(j, i, pos.shape[0])

    idx_i, idx_j, idx_k = gi['idx_i'], gi['idx_j'], gi['idx_k']
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    angle = np.arctan2(np.linalg.norm(np.cross(pos_ji, pos_jk), axis=1),
                       np.sum(pos_ji * pos_jk, axis=1))

    idx_i_t, idx_j_t, idx_k_t, idx_l = gi['idx_i_t'], gi['idx_j_t'], gi['idx_k_t'], gi['idx_l']
    if idx_i_t.shape[0] > 0:
        pos_ij = pos[idx_j_t] - pos[idx_i_t]
        pos_jk_t = pos[idx_k_t] - pos[idx_j_t]
        pos_kl = pos[idx_l] - pos[idx_k_t]
        n1 = np.cross(pos_ij, pos_jk_t)
        n2 = np.cross(pos_jk_t, pos_kl)
        b = np.sum(np.cross(n1, n2) * pos_jk_t, axis=1) / np.linalg.norm(pos_jk_t, axis=1)
        torsion = np.arctan2(b, np.sum(n1 * n2, axis=1))
    else:
        torsion = np.zeros((0,), dtype=np.float64)

    out = {
        'dists': dist.astype(DT),
        'angles': angle.astype(DT),
        'torsions': torsion.astype(DT),
    }
    out.update(gi)
    return out


def radius_graph(pos, cutoff):
    """对应 torch_cluster.radius(pos, pos, cutoff) + remove_self_loops。

    返回 (j, i)：j 为源原子、i 为目标原子（与原实现一样按行分组、去掉自环）。
    """
    pos = np.asarray(pos, dtype=np.float64)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    j, i = np.nonzero(d <= cutoff)
    return remove_self_loops(j.astype(np.int64), i.astype(np.int64))
