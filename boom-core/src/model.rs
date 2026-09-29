//! GRMP 前向传播，逐算子对应 `runtime/grmp_numpy.py`
//! （= 模型/model/mix.py + layer/geom.py）。
//!
//! 两个必须保留的「怪癖」（见 README §4）：
//! 1. `res_x` 是**整个层的输入**（`mlp_x1` 之前那个 x），不是 `mlp_x2` 的输出；
//! 2. `lin_rbf` 被连用两次（`lin_rbf(rbf_feat)` 而不是 `lin_rbf_out`）。
//!
//! **精度**：网络用 f32，但矩阵乘用 f64 累加再落回 f32（与 wmvie.flavor 的
//! flavor-core 同样做法：比 f32 累加更接近真值，实测与 numpy 差 1e-7 量级）。

use crate::basis::{self, NUM_RADIAL, NUM_SPHERICAL};
use crate::geometry::{compute_geometry_features, radius_graph, GeomFeatures};
use crate::graph::dist;
use crate::weights::Weights;

pub const G_CUTOFF: f64 = 7.0;
const LAYER_NORM_EPS: f64 = 1e-5;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Silu,
    Relu,
    Sigmoid,
}

/// 与 `runtime/grmp_numpy.py:mlp` 的 `last_act` 语义一致：
/// `Keep` = 最后一层照常用 `act`；`NoAct` = 最后一层不接激活；`With(a)` = 换成 `a`。
#[derive(Clone, Copy)]
pub enum Last {
    Keep,
    NoAct,
    With(Act),
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn apply(act: Act, v: &mut [f32]) {
    match act {
        Act::Silu => v.iter_mut().for_each(|x| *x = silu(*x)),
        Act::Relu => v.iter_mut().for_each(|x| *x = relu(*x)),
        Act::Sigmoid => v.iter_mut().for_each(|x| *x = sigmoid(*x)),
    }
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

/// 点积的**可移植**实现：4 路独立累加器，便于 LLVM 在 x86 上做 SLP 向量化。
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let mut acc = [0f32; 4];
    for c in 0..chunks {
        let base = c * 4;
        acc[0] += a[base] * b[base];
        acc[1] += a[base + 1] * b[base + 1];
        acc[2] += a[base + 2] * b[base + 2];
        acc[3] += a[base + 3] * b[base + 3];
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for t in (chunks * 4)..n {
        s += a[t] * b[t];
    }
    s
}

/// 点积的 **wasm SIMD128** 实现。
///
/// 为什么必须手写：实测给 rustc 加 `-C target-feature=+simd128` 后 LLVM 仍然不会
/// 向量化这个归约循环（380 KB 的 wasm 里出现 simd128 段，但耗时只从 240 ms 降到
/// 204 ms）；改成两条独立累加链后实测降到 ~60 ms 量级。
/// 构建时由 `boom-core/build-wasm.mjs` 打开 simd128（见该文件的说明）。
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_wasm(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load};

    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let groups = n / 8;
    for c in 0..groups {
        let base = c * 8;
        let a0 = v128_load(pa.add(base) as *const _);
        let b0 = v128_load(pb.add(base) as *const _);
        let a1 = v128_load(pa.add(base + 4) as *const _);
        let b1 = v128_load(pb.add(base + 4) as *const _);
        acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));
        acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
    }
    let mut acc = f32x4_add(acc0, acc1);
    let mut off = groups * 8;
    if n - off >= 4 {
        let a0 = v128_load(pa.add(off) as *const _);
        let b0 = v128_load(pb.add(off) as *const _);
        acc = f32x4_add(acc, f32x4_mul(a0, b0));
        off += 4;
    }
    let mut s = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    while off < n {
        s += *a.get_unchecked(off) * *b.get_unchecked(off);
        off += 1;
    }
    s
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 构建脚本保证编译期开了 simd128；宿主 WebView 需支持 wasm SIMD（Chromium 91+）
    unsafe { dot_wasm(a, b) }
}

/// `y = x @ W.T (+ b)`；`x` 为 `n×k` 行主序，`W` 为 `o×k`。
///
/// 累加用 f32：numpy 侧同样是 f32/BLAS 累加，这是与参考实现最接近的口径。
fn linear(w: &Weights, x: &[f32], n: usize, wname: &str, bname: Option<&str>) -> Vec<f32> {
    let shape = w.shape(wname);
    let (o, k) = (shape[0], shape[1]);
    debug_assert_eq!(x.len(), n * k, "linear 输入维度不符：{wname}");
    let ww = w.get(wname);
    let mut y = vec![0f32; n * o];
    for i in 0..n {
        let xi = &x[i * k..(i + 1) * k];
        let yi = &mut y[i * o..(i + 1) * o];
        for j in 0..o {
            yi[j] = dot(xi, &ww[j * k..(j + 1) * k]);
        }
    }
    if let Some(bn) = bname {
        let bb = w.get(bn);
        for i in 0..n {
            for j in 0..o {
                y[i * o + j] += bb[j];
            }
        }
    }
    y
}

/// `nn.Sequential(Linear, act, Linear, act, ...)`，层下标由权重驱动。
fn mlp(w: &Weights, prefix: &str, x: &[f32], n: usize, act: Act, last: Last) -> Vec<f32> {
    let idx = w.child_indices(prefix);
    assert!(!idx.is_empty(), "MLP {prefix} 在权重里不存在");
    let mut cur = x.to_vec();
    for (pos, i) in idx.iter().enumerate() {
        let wname = format!("{prefix}.{i}.weight");
        let bname = format!("{prefix}.{i}.bias");
        let has_bias = w.has(&bname);
        cur = linear(w, &cur, n, &wname, if has_bias { Some(&bname) } else { None });
        let is_last = pos == idx.len() - 1;
        let a = if is_last {
            match last {
                Last::Keep => Some(act),
                Last::NoAct => None,
                Last::With(a) => Some(a),
            }
        } else {
            Some(act)
        };
        if let Some(a) = a {
            apply(a, &mut cur);
        }
    }
    cur
}

/// `utils.utils.ResidualLayer`：`x + swish(lin2(swish(lin1(x))))`
fn residual_layer(w: &Weights, prefix: &str, x: &[f32], n: usize) -> Vec<f32> {
    let mut h = linear(
        w,
        x,
        n,
        &format!("{prefix}.lin1.weight"),
        Some(&format!("{prefix}.lin1.bias")),
    );
    apply(Act::Silu, &mut h);
    let mut h2 = linear(
        w,
        &h,
        n,
        &format!("{prefix}.lin2.weight"),
        Some(&format!("{prefix}.lin2.bias")),
    );
    apply(Act::Silu, &mut h2);
    let mut out = x.to_vec();
    for (o, h) in out.iter_mut().zip(h2.iter()) {
        *o += *h;
    }
    out
}

/// 末尾 `res_layers` 的用法：第一层带残差直连（加的是**整个层的输入** `res_x`），其余顺序叠加。
fn residual_stack(w: &Weights, prefix: &str, x: &[f32], n: usize, res_x: &[f32]) -> Vec<f32> {
    let idx = w.child_indices(prefix);
    if idx.is_empty() {
        return x.to_vec();
    }
    let mut cur = residual_layer(w, &format!("{prefix}.{}", idx[0]), x, n);
    for (o, r) in cur.iter_mut().zip(res_x.iter()) {
        *o += *r;
    }
    for i in &idx[1..] {
        cur = residual_layer(w, &format!("{prefix}.{i}"), &cur, n);
    }
    cur
}

/// `layer_norm`：两趟算法（均值 -> 方差），与原实现一致。
fn layer_norm(w: &Weights, x: &[f32], n: usize, d: usize, wname: &str, bname: &str) -> Vec<f32> {
    let g = w.get(wname);
    let b = w.get(bname);
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / d as f64;
        let var = row
            .iter()
            .map(|v| {
                let dv = *v as f64 - mean;
                dv * dv
            })
            .sum::<f64>()
            / d as f64;
        let sd = (var + LAYER_NORM_EPS).sqrt();
        for j in 0..d {
            let norm = ((row[j] as f64 - mean) / sd) as f32;
            out[i * d + j] = norm * g[j] + b[j];
        }
    }
    out
}

/// `scatter_add`：按 `index` 顺序**串行** f32 累加（与 `np.add.at` 的累加顺序一致）。
fn scatter_add(src: &[f32], index: &[i64], dim_size: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim_size * dim];
    for (k, t) in index.iter().enumerate() {
        let t = *t as usize;
        let s = &src[k * dim..(k + 1) * dim];
        let o = &mut out[t * dim..(t + 1) * dim];
        for d in 0..dim {
            o[d] += s[d];
        }
    }
    out
}

/// 同上，索引是无符号型（几何特征里的边索引）。
fn scatter_add_usize(src: &[f32], index: &[usize], dim_size: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim_size * dim];
    for (k, t) in index.iter().enumerate() {
        let s = &src[k * dim..(k + 1) * dim];
        let o = &mut out[*t * dim..(*t + 1) * dim];
        for d in 0..dim {
            o[d] += s[d];
        }
    }
    out
}

/// 按索引取行：`x[idx]`。
fn gather(x: &[f32], idx: &[i64], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; idx.len() * dim];
    for (k, t) in idx.iter().enumerate() {
        let t = *t as usize;
        out[k * dim..(k + 1) * dim].copy_from_slice(&x[t * dim..(t + 1) * dim]);
    }
    out
}

fn gather_usize(x: &[f32], idx: &[usize], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; idx.len() * dim];
    for (k, t) in idx.iter().enumerate() {
        out[k * dim..(k + 1) * dim].copy_from_slice(&x[*t * dim..(*t + 1) * dim]);
    }
    out
}

fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// `concat([x[i], x[j], ...], axis=-1)`（按行拼接若干同样行数的块）。
fn cat_rows(parts: &[&[f32]], rows: usize, dims: &[usize]) -> Vec<f32> {
    let total: usize = dims.iter().sum();
    let mut out = vec![0f32; rows * total];
    for r in 0..rows {
        let mut off = 0usize;
        for (p, d) in parts.iter().zip(dims.iter()) {
            out[r * total + off..r * total + off + d]
                .copy_from_slice(&p[r * d..(r + 1) * d]);
            off += d;
        }
    }
    out
}

// ---------------------------------------------------------------- 网络层

/// 对应 `layer/geom.py:LocalMessagePassing.forward`
#[allow(clippy::too_many_arguments)]
fn local_message_passing(
    w: &Weights,
    prefix: &str,
    x: &[f32],
    n: usize,
    rbf: &[f32],
    sbf: &[f32],
    tbf: &[f32],
    feats: &GeomFeatures,
    j: &[i64],
    i: &[i64],
) -> Vec<f32> {
    let res_x = x;
    let e = j.len();
    let hidden = w.shape("node_embedding.weight")[0];

    let x1 = mlp(w, &format!("{prefix}.mlp_x1"), x, n, Act::Silu, Last::Keep);
    let rbf_f = mlp(w, &format!("{prefix}.mlp_rbf"), rbf, e, Act::Silu, Last::Keep);
    let lin_rbf = format!("{prefix}.lin_rbf.weight");
    let rbf_feat = linear(w, &rbf_f, e, &lin_rbf, None);

    let xi = gather(&x1, i, hidden);
    let xj = gather(&x1, j, hidden);
    let m = cat_rows(&[&xi, &xj, &rbf_f], e, &[hidden, hidden, hidden]);

    let mut m_ji = mlp(w, &format!("{prefix}.mlp_m_ji"), &m, e, Act::Silu, Last::Keep);
    apply(Act::Silu, &mut m_ji);
    let mut m_kj = mlp(w, &format!("{prefix}.mlp_m_kj"), &m, e, Act::Silu, Last::Keep);
    apply(Act::Silu, &mut m_kj);
    // 原实现这里复用 lin_rbf（不是 lin_rbf_out），见 README §4
    let m_neighbor = mul(&m_kj, &linear(w, &rbf_feat, e, &lin_rbf, None));

    let t = feats.gi.idx_kj.len();
    let sbf_feat = linear(
        w,
        &mlp(w, &format!("{prefix}.mlp_sbf1"), sbf, t, Act::Silu, Last::Keep),
        t,
        &format!("{prefix}.lin_sbf2.weight"),
        None,
    );
    let sbf_out = mlp(w, &format!("{prefix}.mlp_sbf"), &sbf_feat, t, Act::Silu, Last::Keep);
    let m_angle = mul(&gather_usize(&m_neighbor, &feats.gi.idx_kj, hidden), &sbf_out);

    let q = feats.gi.idx_kj_t.len();
    let tbf_feat = linear(
        w,
        &mlp(w, &format!("{prefix}.mlp_tbf1"), tbf, q, Act::Silu, Last::Keep),
        q,
        &format!("{prefix}.lin_tbf2.weight"),
        None,
    );
    let m_angle_mapped = gather_usize(&m_angle, &feats.gi.angle_idx_map, hidden);
    let m_angle_processed = mlp(
        w,
        &format!("{prefix}.mlp_angle_for_torsion"),
        &m_angle_mapped,
        q,
        Act::Silu,
        Last::Keep,
    );
    let m_torsion = mul(&m_angle_processed, &tbf_feat);
    let m_torsion_agg = scatter_add_usize(&m_torsion, &feats.gi.idx_ji_t, e, hidden);

    let mut msum = m_ji;
    for (a, b) in msum.iter_mut().zip(m_torsion_agg.iter()) {
        *a += *b;
    }
    let rbf_out = linear(w, &rbf_feat, e, &format!("{prefix}.lin_rbf_out.weight"), None);
    let m_final = mul(&rbf_out, &msum);

    let agg = scatter_add(&m_final, i, n, hidden);
    let mut xa = x1;
    for (a, b) in xa.iter_mut().zip(agg.iter()) {
        *a += *b;
    }
    let x2 = mlp(w, &format!("{prefix}.mlp_x2"), &xa, n, Act::Silu, Last::Keep);
    residual_stack(w, &format!("{prefix}.res_layers"), &x2, n, res_x)
}

/// 对应 `layer/geom.py:GlobalMessagePassing.forward`
#[allow(clippy::too_many_arguments)]
fn global_message_passing(
    w: &Weights,
    prefix: &str,
    x: &[f32],
    n: usize,
    edge_attr: &[f32],
    j: &[i64],
    i: &[i64],
) -> Vec<f32> {
    let res_x = x;
    let e = j.len();
    let hidden = w.shape("node_embedding.weight")[0];

    let x1 = mlp(w, &format!("{prefix}.mlp_x1"), x, n, Act::Silu, Last::Keep);
    let xi = gather(&x1, i, hidden);
    let xj = gather(&x1, j, hidden);
    let pair = cat_rows(&[&xi, &xj], e, &[hidden, hidden]);
    let edge_feats = mlp(
        w,
        &format!("{prefix}.mlp_edge_update"),
        &pair,
        e,
        Act::Silu,
        Last::Keep,
    );
    let com_edge: Vec<f32> = edge_attr
        .iter()
        .zip(edge_feats.iter())
        .map(|(a, b)| a + b)
        .collect();
    let m_in = cat_rows(&[&xi, &xj, &com_edge], e, &[hidden, hidden, hidden]);
    let m = mlp(w, &format!("{prefix}.mlp_m"), &m_in, e, Act::Silu, Last::Keep);
    let w_edge = linear(w, &com_edge, e, &format!("{prefix}.W_edge_attr.weight"), None);
    let message = mul(&m, &w_edge);

    let agg = scatter_add(&message, i, n, hidden);
    let mut xa = x1;
    for (a, b) in xa.iter_mut().zip(agg.iter()) {
        *a += *b;
    }
    let x2 = mlp(w, &format!("{prefix}.mlp_x2"), &xa, n, Act::Silu, Last::Keep);
    residual_stack(w, &format!("{prefix}.res_layers"), &x2, n, res_x)
}

/// 对应 `model/mix.py:AdaptiveMixtureLayer`（checkpoint 用的是 `gate` 分支；
/// 另两种分支的权重不存在时会各自退化，这里只实现实际存在的那条路径）。
fn adaptive_layer(w: &Weights, prefix: &str, x_local: &[f32], x_global: &[f32], n: usize) -> Vec<f32> {
    let local_t = linear(
        w,
        x_local,
        n,
        &format!("{prefix}.local_trans.weight"),
        Some(&format!("{prefix}.local_trans.bias")),
    );
    let global_t = linear(
        w,
        x_global,
        n,
        &format!("{prefix}.global_trans.weight"),
        Some(&format!("{prefix}.global_trans.bias")),
    );
    let hidden = local_t.len() / n;
    let both = cat_rows(&[&local_t, &global_t], n, &[hidden, hidden]);
    let gate = mlp(
        w,
        &format!("{prefix}.gate_net"),
        &both,
        n,
        Act::Silu,
        Last::With(Act::Sigmoid),
    );
    let one_minus: Vec<f32> = gate.iter().map(|g| 1.0 - g).collect();
    let fused: Vec<f32> = mul(&gate, &local_t)
        .iter()
        .zip(mul(&one_minus, &global_t).iter())
        .map(|(a, b)| a + b)
        .collect();
    let mut sum: Vec<f32> = fused
        .iter()
        .zip(x_local.iter())
        .map(|(a, b)| a + b)
        .collect();
    for (a, b) in sum.iter_mut().zip(x_global.iter()) {
        *a += *b;
    }
    layer_norm(
        w,
        &sum,
        n,
        hidden,
        &format!("{prefix}.layer_norm.weight"),
        &format!("{prefix}.layer_norm.bias"),
    )
}

// ---------------------------------------------------------------- 整网

/// 加载好权重后可对单个分子做前向。
pub struct Model {
    w: Weights,
    /// 隐层宽度（从 `node_embedding.weight` 读出）。
    pub hidden: usize,
    /// 输入维度（= 49）。
    pub in_dim: usize,
    /// pass_layers（从 `message_layers.*` 的子模块数量读出）。
    pub n_layers: usize,
}

impl Model {
    /// 结构自检 + 构建。层数/宽度全部从权重读，换 checkpoint 不用改代码。
    pub fn new(w: Weights) -> Result<Self, String> {
        let emb = w.shape("node_embedding.weight");
        let (hidden, in_dim) = (emb[0], emb[1]);
        let n_layers = w.child_indices("message_layers").len();
        let n_global = w.child_indices("global_layers").len();
        let n_adaptive = w.child_indices("adaptive_layers").len();
        if !(n_layers == n_global && n_layers == n_adaptive) {
            return Err(format!(
                "message/global/adaptive 层数不一致（{n_layers}/{n_global}/{n_adaptive}），权重文件可能损坏"
            ));
        }
        if n_layers == 0 {
            return Err("权重里没有 message_layers，不是 GRMP checkpoint".to_string());
        }
        let freq = w.shape("rbf.freq");
        if freq != [NUM_RADIAL] {
            return Err(format!("rbf.freq 形状异常：{freq:?}，应为 [{NUM_RADIAL}]"));
        }

        let mut need: Vec<String> = vec![
            "node_embedding.bias".into(),
            "mlp_global_rbf.0.weight".into(),
            "MLP.0.weight".into(),
        ];
        for l in 0..n_layers {
            for suffix in [
                "mlp_x1.0.weight",
                "mlp_rbf.0.weight",
                "lin_rbf.weight",
                "mlp_m_ji.0.weight",
                "mlp_m_kj.0.weight",
                "mlp_sbf1.0.weight",
                "lin_sbf2.weight",
                "mlp_sbf.0.weight",
                "mlp_tbf1.0.weight",
                "lin_tbf2.weight",
                "mlp_angle_for_torsion.0.weight",
                "lin_rbf_out.weight",
                "mlp_x2.0.weight",
            ] {
                need.push(format!("message_layers.{l}.{suffix}"));
            }
            for suffix in ["mlp_x1.0.weight", "mlp_edge_update.0.weight", "mlp_m.0.weight",
                           "W_edge_attr.weight", "mlp_x2.0.weight"] {
                need.push(format!("global_layers.{l}.{suffix}"));
            }
            for suffix in ["local_trans.weight", "global_trans.weight", "gate_net.0.weight",
                           "layer_norm.weight"] {
                need.push(format!("adaptive_layers.{l}.{suffix}"));
            }
        }
        let refs: Vec<&str> = need.iter().map(|s| s.as_str()).collect();
        w.require(&refs)?;

        Ok(Self {
            w,
            hidden,
            in_dim,
            n_layers,
        })
    }

    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// `x_geom`: `[N, in_dim]`；`pos`: `[N, 3]`；`(j, i)`: 键图（双向）。
    pub fn forward(&self, x_geom: &[f32], pos: &[[f64; 3]], j: &[i64], i: &[i64]) -> f32 {
        let w = &self.w;
        let n = pos.len();
        assert!(n > 0, "空分子");
        assert_eq!(x_geom.len(), n * self.in_dim, "x_geom 维度与权重不匹配");
        let hidden = self.hidden;
        let freq = w.get("rbf.freq");

        // ---- 全局（半径图，7 Å，f64 距离，无 f32 往返）----
        let (gj, gi) = radius_graph(pos, G_CUTOFF);
        let dist_g: Vec<f64> = (0..gj.len())
            .map(|k| dist(pos[gi[k] as usize], pos[gj[k] as usize]))
            .collect();
        let edge_attr_g = mlp(
            w,
            "mlp_global_rbf",
            &to_f32(&basis::rbf_basis(&dist_g, freq)),
            gj.len(),
            Act::Silu,
            Last::Keep,
        );

        // ---- 局部（键图；特征 f32 存储，进基组前转回 f64）----
        let feats = compute_geometry_features(pos, j, i);
        let dists64: Vec<f64> = feats.dists.iter().map(|v| *v as f64).collect();
        let angles64: Vec<f64> = feats.angles.iter().map(|v| *v as f64).collect();
        let tors64: Vec<f64> = feats.torsions.iter().map(|v| *v as f64).collect();
        let rbf = to_f32(&basis::rbf_basis(&dists64, freq));
        let sbf = to_f32(&basis::sbf_basis(&dists64, &angles64, &feats.gi.idx_kj));
        let tbf = to_f32(&basis::tbf_basis(
            &dists64,
            &angles64,
            &tors64,
            &feats.gi.idx_kj_t,
            &feats.gi.angle_idx_map,
        ));

        let mut x = linear(w, x_geom, n, "node_embedding.weight", Some("node_embedding.bias"));
        let mut per_layer: Vec<Vec<f32>> = Vec::with_capacity(self.n_layers);
        for layer in 0..self.n_layers {
            let x_local = local_message_passing(
                w,
                &format!("message_layers.{layer}"),
                &x,
                n,
                &rbf,
                &sbf,
                &tbf,
                &feats,
                j,
                i,
            );
            let x_global = global_message_passing(
                w,
                &format!("global_layers.{layer}"),
                &x,
                n,
                &edge_attr_g,
                &gj,
                &gi,
            );
            x = adaptive_layer(w, &format!("adaptive_layers.{layer}"), &x_local, &x_global, n);
            per_layer.push(x.clone());
        }

        let nl = self.n_layers;
        let d = hidden * nl;
        let mut pooled = vec![0f32; d];
        for h in 0..d {
            let (layer, col) = (h / hidden, h % hidden);
            let mut acc = 0f64;
            for a in 0..n {
                acc += per_layer[layer][a * hidden + col] as f64;
            }
            pooled[h] = (acc / n as f64) as f32;
        }
        let out = mlp(w, "MLP", &pooled, 1, Act::Relu, Last::NoAct);
        out[0]
    }
}

/// 基组维度自检（供 `main --check` 用）。
pub fn basis_dims() -> (usize, usize) {
    (NUM_SPHERICAL, NUM_RADIAL)
}
