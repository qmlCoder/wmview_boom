"""GRMP 前向传播的纯 numpy 实现（权重由 tools/export_weights.py 从 .pt 转换）。

逐行对齐 模型/model/mix.py + 模型/layer/geom.py + 模型/utils/{utils,special}.py：
结构由权重里的 key 驱动（层数、MLP 宽度、融合方式都从 npz 里读），
所以换 checkpoint（例如原项目后续重新训练的版本）只要模块结构不变就能直接用。

数值约定：全程 float32，和原 PyTorch 模型（fp32）保持一致。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import basis_gen as B  # noqa: E402
from geometry import compute_geometry_features, radius_graph  # noqa: E402

DT = np.float32
NUM_SPHERICAL, NUM_RADIAL = B.NUM_SPHERICAL, B.NUM_RADIAL
CUTOFF = 5.0          # 局部（键图）距离截断
G_CUTOFF = 7.0        # 全局（半径图）截断
ENVELOPE_EXPONENT = 5


# ---------------------------------------------------------------- 基础算子

def sigmoid(x):
    x = np.asarray(x, dtype=DT)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def silu(x):
    """x * sigmoid(x)（原代码里叫 swish / SiLU）。"""
    return x * sigmoid(x)


swish = silu


def relu(x):
    return np.maximum(x, 0)


def linear(x, w, b=None):
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def scatter_add(src, index, dim_size):
    if src.shape[0] == 0:
        return np.zeros((dim_size, src.shape[1]), dtype=DT)
    out = np.zeros((dim_size, src.shape[1]), dtype=DT)
    np.add.at(out, index, src)
    return out


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


_KEEP = object()


def _child_indices(weights, prefix):
    out = []
    head = prefix + '.'
    for k in weights:
        if k.startswith(head) and k.endswith('.weight'):
            rest = k[len(head):]
            first = rest.split('.')[0]
            if first.isdigit():
                out.append(int(first))
    return sorted(set(out))


def mlp(weights, prefix, x, act=silu, last_act=_KEEP):
    """nn.Sequential(Linear, act, Linear, act, ...)：每层 Linear 后接一个激活。

    last_act 用来覆盖最后一层的激活（None 表示最后一层不接激活）。
    """
    idx = _child_indices(weights, prefix)
    for n, i in enumerate(idx):
        x = linear(x, weights[f'{prefix}.{i}.weight'], weights.get(f'{prefix}.{i}.bias'))
        if n == len(idx) - 1 and last_act is not _KEEP:
            if last_act is not None:
                x = last_act(x)
        else:
            x = act(x)
    return x


def residual_layer(weights, prefix, x):
    """utils.utils.ResidualLayer：x + swish(lin2(swish(lin1(x))))"""
    h = linear(x, weights[f'{prefix}.lin1.weight'], weights[f'{prefix}.lin1.bias'])
    h = linear(swish(h), weights[f'{prefix}.lin2.weight'], weights[f'{prefix}.lin2.bias'])
    return x + swish(h)


def residual_stack(weights, prefix, x, res_x):
    """末尾 res_layers 的用法：第一层带残差直连，其余顺序叠加。

    注意 res_x 是**整个层的输入**（mlp_x1 之前那个 x），不是 mlp_x2 的输出。
    """
    idx = _child_indices(weights, prefix)
    if not idx:
        return x
    x = residual_layer(weights, f'{prefix}.{idx[0]}', x) + res_x
    for i in idx[1:]:
        x = residual_layer(weights, f'{prefix}.{i}', x)
    return x


# ---------------------------------------------------------------- 几何基组

def envelope(x, p=ENVELOPE_EXPONENT):
    a = -(p + 1) * (p + 2) / 2.0
    b = p * (p + 2)
    c = -p * (p + 1) / 2.0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        val = 1.0 / x + a * x ** p + b * x ** (p + 1) + c * x ** (p + 2)
    return np.where(x < 1, val, 0.0)


def rbf_basis(dists, freq):
    """BesselBasisLayer：envelope(d/c) * sin(freq * d/c)（freq 是可学习参数）。"""
    rs = np.asarray(dists, dtype=np.float64) / CUTOFF
    env = envelope(rs)[:, None]
    return (env * np.sin(np.asarray(freq, dtype=np.float64)[None, :] * rs[:, None])).astype(DT)


def sbf_basis(dists, angles, idx_kj):
    """SphericalBasisLayer：[num_angles, 3*6]"""
    rs = np.asarray(dists, dtype=np.float64) / CUTOFF
    rbf = B.sbf_radial(rs) * envelope(rs)[:, None]
    cbf = B.sph_sbf(np.asarray(angles, dtype=np.float64))
    n, k = NUM_SPHERICAL, NUM_RADIAL
    sel = rbf[idx_kj].reshape(-1, n, k)
    return (sel * cbf.reshape(-1, n, 1)).reshape(-1, n * k).astype(DT)


def tbf_basis(dists, angles, torsions, idx_kj_t, angle_idx_map):
    """torsionLayer：[num_torsions, 3*3*6]（注意没有 envelope）"""
    rs = np.asarray(dists, dtype=np.float64) / CUTOFF
    rbf = B.sbf_radial(rs)
    angle_for_torsion = np.asarray(angles, dtype=np.float64)[angle_idx_map]
    cbf = B.sph_tbf(angle_for_torsion, np.asarray(torsions, dtype=np.float64))
    n, k = NUM_SPHERICAL, NUM_RADIAL
    sel = rbf[idx_kj_t].reshape(-1, n, k)
    out = sel[:, None, :, :] * cbf.reshape(-1, n, n)[:, :, :, None]
    return out.reshape(-1, n * n * k).astype(DT)


# ---------------------------------------------------------------- 网络层

def local_message_passing(w, prefix, x, rbf, sbf, tbf, feats, j, i):
    """对应 layer/geom.py:LocalMessagePassing.forward"""
    res_x = x
    n_nodes = x.shape[0]
    x = mlp(w, f'{prefix}.mlp_x1', x)

    rbf_f = mlp(w, f'{prefix}.mlp_rbf', rbf)
    rbf_feat = linear(rbf_f, w[f'{prefix}.lin_rbf.weight'])
    m = np.concatenate([x[i], x[j], rbf_f], axis=-1)
    m_ji = swish(mlp(w, f'{prefix}.mlp_m_ji', m))
    m_kj = swish(mlp(w, f'{prefix}.mlp_m_kj', m))
    # 原代码这里是 lin_rbf(rbf_feat)：同一个线性层连用了两次（见 模型/README.md §6.2）
    m_neighbor = m_kj * linear(rbf_feat, w[f'{prefix}.lin_rbf.weight'])

    sbf_feat = linear(mlp(w, f'{prefix}.mlp_sbf1', sbf), w[f'{prefix}.lin_sbf2.weight'])
    m_angle = m_neighbor[feats['idx_kj']] * mlp(w, f'{prefix}.mlp_sbf', sbf_feat)

    tbf_feat = linear(mlp(w, f'{prefix}.mlp_tbf1', tbf), w[f'{prefix}.lin_tbf2.weight'])
    m_angle_mapped = m_angle[feats['angle_idx_map']]
    m_angle_processed = mlp(w, f'{prefix}.mlp_angle_for_torsion', m_angle_mapped)
    m_torsion = m_angle_processed * tbf_feat
    m_torsion_agg = scatter_add(m_torsion, feats['idx_ji_t'], m.shape[0])

    m = m_ji + m_torsion_agg
    m_final = linear(rbf_feat, w[f'{prefix}.lin_rbf_out.weight']) * m
    x = x + scatter_add(m_final, i, n_nodes)
    x = mlp(w, f'{prefix}.mlp_x2', x)
    return residual_stack(w, f'{prefix}.res_layers', x, res_x)


def global_message_passing(w, prefix, x, edge_attr, j, i):
    """对应 layer/geom.py:GlobalMessagePassing.forward"""
    res_x = x
    n_nodes = x.shape[0]
    x = mlp(w, f'{prefix}.mlp_x1', x)
    edge_feats = mlp(w, f'{prefix}.mlp_edge_update', np.concatenate([x[i], x[j]], axis=-1))
    com_edge = edge_attr + edge_feats
    m = mlp(w, f'{prefix}.mlp_m', np.concatenate([x[i], x[j], com_edge], axis=-1))
    message = m * linear(com_edge, w[f'{prefix}.W_edge_attr.weight'])
    x = x + scatter_add(message, i, n_nodes)
    x = mlp(w, f'{prefix}.mlp_x2', x)
    return residual_stack(w, f'{prefix}.res_layers', x, res_x)


def adaptive_layer(w, prefix, x_local, x_global):
    """对应 model/mix.py:AdaptiveMixtureLayer（三种融合方式都支持）。"""
    local_t = linear(x_local, w[prefix + '.local_trans.weight'], w[prefix + '.local_trans.bias'])
    global_t = linear(x_global, w[prefix + '.global_trans.weight'], w[prefix + '.global_trans.bias'])

    if f'{prefix}.gate_net.0.weight' in w:
        gate = mlp(w, f'{prefix}.gate_net', np.concatenate([local_t, global_t], axis=-1),
                   act=silu, last_act=sigmoid)
        fused = gate * local_t + (1.0 - gate) * global_t
    elif f'{prefix}.ratio' in w:
        ratio = w[prefix + '.ratio']
        p = np.exp(ratio - ratio.max())
        p = p / p.sum()
        fused = p[0] * local_t + p[1] * global_t
    elif f'{prefix}.attention_net.0.weight' in w:
        combined = np.concatenate([local_t, global_t], axis=-1)
        att = mlp(w, f'{prefix}.attention_net', combined, act=silu)
        p = np.exp(att - att.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        fused = p[..., 0:1] * local_t + p[..., 1:2] * global_t
    else:
        fused = local_t + global_t

    return layer_norm(fused + x_local + x_global,
                      w[prefix + '.layer_norm.weight'], w[prefix + '.layer_norm.bias'])


# ---------------------------------------------------------------- 整网

class GRMP:
    """加载 npz 权重后可直接对单个分子做前向。"""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.w = {k: np.asarray(data[k], dtype=DT) for k in data.files}
        self.n_layers = len(_child_indices(self.w, 'message_layers'))
        self.n_global = len(_child_indices(self.w, 'global_layers'))
        self.n_adaptive = len(_child_indices(self.w, 'adaptive_layers'))
        if not (self.n_layers == self.n_global == self.n_adaptive):
            raise ValueError('message/global/adaptive 层数不一致，权重文件可能损坏')

    def forward(self, x_geom, pos, edge_index):
        """x_geom: [N,49]；pos: [N,3]；edge_index: (j, i) 键图（双向）。"""
        w = self.w
        x_geom = np.asarray(x_geom, dtype=DT)
        pos = np.asarray(pos, dtype=np.float64)
        j = np.asarray(edge_index[0], dtype=np.int64)
        i = np.asarray(edge_index[1], dtype=np.int64)

        gj, gi = radius_graph(pos, G_CUTOFF)
        dist_g = np.linalg.norm(pos[gi] - pos[gj], axis=1)
        edge_attr_g = mlp(w, 'mlp_global_rbf', rbf_basis(dist_g, w['rbf.freq']))

        feats = compute_geometry_features(pos, j, i)
        rbf = rbf_basis(feats['dists'], w['rbf.freq'])
        sbf = sbf_basis(feats['dists'], feats['angles'], feats['idx_kj'])
        tbf = tbf_basis(feats['dists'], feats['angles'], feats['torsions'],
                        feats['idx_kj_t'], feats['angle_idx_map'])

        x = linear(x_geom, w['node_embedding.weight'], w['node_embedding.bias'])
        per_layer = []
        for layer in range(self.n_layers):
            x_local = local_message_passing(w, f'message_layers.{layer}', x, rbf, sbf, tbf, feats, j, i)
            x_global = global_message_passing(w, f'global_layers.{layer}', x, edge_attr_g, gj, gi)
            x = adaptive_layer(w, f'adaptive_layers.{layer}', x_local, x_global)
            per_layer.append(x)

        geom = np.concatenate(per_layer, axis=-1)
        pooled = geom.mean(axis=0, keepdims=True)            # global_mean_pool
        return mlp(w, 'MLP', pooled, act=relu, last_act=None)


def predict(npz_path, x_geom, pos, edge_index):
    return float(GRMP(npz_path).forward(x_geom, pos, edge_index)[0, 0])
