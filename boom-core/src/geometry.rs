//! 图 / 几何特征，逐行对应 `runtime/geometry.py`（= 模型/utils/special.py + layer/geom.py）。
//!
//! 原实现依赖 `torch_sparse.SparseTensor` 做「按目标节点取边」，这里用等价的索引实现
//! （顺序语义一致：保持边的输入顺序；`np.argsort(kind='stable')` 换成计数分桶，天然稳定）。
//!
//! **精度**：距离/键角/二面角全程 f64，最后才 `.astype(f32)`——必须照抄，
//! 否则会把 1e-6 级的对拍误差放大到 1e-4 以上。

use crate::graph::dist;

/// `by_target[t]` = 「终点为 t」的边索引，保持边序（等价 `torch_sparse` 的 `adj_t[t]`）。
fn edges_by_target(i: &[i64], num_nodes: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); num_nodes];
    for (k, t) in i.iter().enumerate() {
        let t = *t;
        if t >= 0 && (t as usize) < num_nodes {
            out[t as usize].push(k);
        }
    }
    out
}

/// 三体 `(i-j-k)` 与四体 `(i-j-k-l)` 索引，以及「四体 -> 三体」映射。
/// 对应 `utils/special.py:get_geometry_indices`。
pub struct GeomIndices {
    pub idx_i: Vec<i64>,
    pub idx_j: Vec<i64>,
    pub idx_k: Vec<i64>,
    pub idx_kj: Vec<usize>,
    pub idx_ji: Vec<usize>,
    pub idx_i_t: Vec<i64>,
    pub idx_j_t: Vec<i64>,
    pub idx_k_t: Vec<i64>,
    pub idx_l: Vec<i64>,
    pub idx_kj_t: Vec<usize>,
    pub idx_ji_t: Vec<usize>,
    pub idx_kl_t: Vec<usize>,
    pub angle_idx_map: Vec<usize>,
}

pub fn geometry_indices(j: &[i64], i: &[i64], num_nodes: usize) -> GeomIndices {
    let e = j.len();
    let by_target = edges_by_target(i, num_nodes);

    // ---- i-j-k ----
    let triple_counts: Vec<usize> = (0..e)
        .map(|k| by_target[j[k] as usize].len())
        .collect();
    let mut idx_i: Vec<i64> = Vec::new();
    let mut idx_j: Vec<i64> = Vec::new();
    let mut idx_ji: Vec<usize> = Vec::new();
    let mut idx_k: Vec<i64> = Vec::new();
    let mut idx_kj: Vec<usize> = Vec::new();
    for k in 0..e {
        for _ in 0..triple_counts[k] {
            idx_i.push(i[k]);
            idx_j.push(j[k]);
            idx_ji.push(k);
        }
        for &ekj in &by_target[j[k] as usize] {
            idx_k.push(j[ekj]);
            idx_kj.push(ekj);
        }
    }
    // mask = idx_i != idx_k
    let mut f_i: Vec<i64> = Vec::new();
    let mut f_j: Vec<i64> = Vec::new();
    let mut f_k: Vec<i64> = Vec::new();
    let mut f_kj: Vec<usize> = Vec::new();
    let mut f_ji: Vec<usize> = Vec::new();
    for t in 0..idx_i.len() {
        if idx_i[t] != idx_k[t] {
            f_i.push(idx_i[t]);
            f_j.push(idx_j[t]);
            f_k.push(idx_k[t]);
            f_kj.push(idx_kj[t]);
            f_ji.push(idx_ji[t]);
        }
    }
    let (idx_i, idx_j, idx_k, idx_kj, idx_ji) = (f_i, f_j, f_k, f_kj, f_ji);

    // ---- i-j-k-l ----
    let quad_counts: Vec<usize> = idx_k
        .iter()
        .map(|k| by_target[*k as usize].len())
        .collect();
    let mut idx_i_t: Vec<i64> = Vec::new();
    let mut idx_j_t: Vec<i64> = Vec::new();
    let mut idx_k_t: Vec<i64> = Vec::new();
    let mut idx_l: Vec<i64> = Vec::new();
    let mut edge_kl: Vec<usize> = Vec::new();
    for t in 0..idx_k.len() {
        let k = idx_k[t] as usize;
        for _ in 0..quad_counts[t] {
            idx_i_t.push(idx_i[t]);
            idx_j_t.push(idx_j[t]);
            idx_k_t.push(idx_k[t]);
        }
        for &ekl in &by_target[k] {
            idx_l.push(j[ekl]);
            edge_kl.push(ekl);
        }
    }
    // mask_q = (idx_l != idx_i_t) & (idx_l != idx_j_t)
    let mut g_i: Vec<i64> = Vec::new();
    let mut g_j: Vec<i64> = Vec::new();
    let mut g_k: Vec<i64> = Vec::new();
    let mut g_l: Vec<i64> = Vec::new();
    let mut g_kj: Vec<usize> = Vec::new();
    let mut g_ji: Vec<usize> = Vec::new();
    let mut g_kl: Vec<usize> = Vec::new();
    for t in 0..idx_i_t.len() {
        if idx_l[t] != idx_i_t[t] && idx_l[t] != idx_j_t[t] {
            g_i.push(idx_i_t[t]);
            g_j.push(idx_j_t[t]);
            g_k.push(idx_k_t[t]);
            g_l.push(idx_l[t]);
            g_kl.push(edge_kl[t]);
        }
    }
    // np.repeat(idx_kj, quad_counts)[mask_q]：展开后再按同一个 mask 过滤
    let mut g_kj_full: Vec<usize> = Vec::new();
    let mut g_ji_full: Vec<usize> = Vec::new();
    for t in 0..idx_kj.len() {
        for _ in 0..quad_counts[t] {
            g_kj_full.push(idx_kj[t]);
            g_ji_full.push(idx_ji[t]);
        }
    }
    let mut g_kjv: Vec<usize> = Vec::new();
    let mut g_jiv: Vec<usize> = Vec::new();
    let mut m = 0usize;
    for t in 0..idx_i_t.len() {
        if idx_l[t] != idx_i_t[t] && idx_l[t] != idx_j_t[t] {
            g_kjv.push(g_kj_full[m]);
            g_jiv.push(g_ji_full[m]);
        }
        m += 1;
    }

    // 四体 -> 三体 的映射：等价 match.max(dim=1)[1]（全 False 时取 0）
    let m_len = idx_i.len();
    let mut angle_idx_map = vec![0usize; g_i.len()];
    for q in 0..g_i.len() {
        for a in 0..m_len {
            if g_i[q] == idx_i[a] && g_j[q] == idx_j[a] && g_k[q] == idx_k[a] {
                angle_idx_map[q] = a;
                break;
            }
        }
    }

    GeomIndices {
        idx_i,
        idx_j,
        idx_k,
        idx_kj,
        idx_ji,
        idx_i_t: g_i,
        idx_j_t: g_j,
        idx_k_t: g_k,
        idx_l: g_l,
        idx_kj_t: g_kjv,
        idx_ji_t: g_jiv,
        idx_kl_t: g_kl,
        angle_idx_map,
    }
}

/// 几何特征：`dists`/`angles`/`torsions` 是 **f32**（与原实现 `.astype(DT)` 一致），
/// 内部计算全程 f64。
pub struct GeomFeatures {
    pub dists: Vec<f32>,
    pub angles: Vec<f32>,
    pub torsions: Vec<f32>,
    pub gi: GeomIndices,
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 对应 `utils/special.py:compute_geometry_features`（输入是键图 edge_index=(j,i)）。
pub fn compute_geometry_features(pos: &[[f64; 3]], j: &[i64], i: &[i64]) -> GeomFeatures {
    let e = j.len();
    let mut dists = Vec::with_capacity(e);
    for t in 0..e {
        dists.push(dist(pos[i[t] as usize], pos[j[t] as usize]) as f32);
    }

    let gi = geometry_indices(j, i, pos.len());

    let mut angles: Vec<f32> = Vec::with_capacity(gi.idx_i.len());
    for t in 0..gi.idx_i.len() {
        let pj = pos[gi.idx_j[t] as usize];
        let pos_ji = sub(pos[gi.idx_i[t] as usize], pj);
        let pos_jk = sub(pos[gi.idx_k[t] as usize], pj);
        let angle = norm(cross(pos_ji, pos_jk)).atan2(dot(pos_ji, pos_jk));
        angles.push(angle as f32);
    }

    let n_t = gi.idx_i_t.len();
    let mut torsions: Vec<f32> = Vec::with_capacity(n_t);
    for t in 0..n_t {
        let pi = pos[gi.idx_i_t[t] as usize];
        let pj = pos[gi.idx_j_t[t] as usize];
        let pk = pos[gi.idx_k_t[t] as usize];
        let pl = pos[gi.idx_l[t] as usize];
        let pos_ij = sub(pj, pi);
        let pos_jk = sub(pk, pj);
        let pos_kl = sub(pl, pk);
        let n1 = cross(pos_ij, pos_jk);
        let n2 = cross(pos_jk, pos_kl);
        let b = dot(cross(n1, n2), pos_jk) / norm(pos_jk);
        let torsion = b.atan2(dot(n1, n2));
        torsions.push(torsion as f32);
    }

    GeomFeatures {
        dists,
        angles,
        torsions,
        gi,
    }
}

/// 对应 `torch_cluster.radius(pos, pos, cutoff)` + `remove_self_loops`。
/// 返回 `(j, i)`：j 为源原子、i 为目标原子，顺序与 `np.nonzero` 的行主序一致。
pub fn radius_graph(pos: &[[f64; 3]], cutoff: f64) -> (Vec<i64>, Vec<i64>) {
    let n = pos.len();
    let mut js: Vec<i64> = Vec::new();
    let mut is: Vec<i64> = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            if dist(pos[a], pos[b]) <= cutoff {
                js.push(a as i64);
                is.push(b as i64);
            }
        }
    }
    (js, is)
}
