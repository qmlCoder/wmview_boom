"""最小化的 torch_cluster / torch_scatter / torch_sparse / torch_geometric 替身。

只为了让**原始模型代码**（模型/model/mix.py、layer/geom.py、utils/special.py）能在
没有那些重依赖的环境里跑起来做数值对齐验证。语义按原库文档实现，不追求完整。
"""

from __future__ import annotations

import sys
import types

import torch


def scatter(src, index, dim=0, dim_size=None, reduce='sum'):
    assert dim == 0
    n = int(index.max()) + 1 if dim_size is None else int(dim_size)
    out = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    if src.numel() == 0:
        return out
    return out.index_add_(0, index, src)


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=64):
    """暴力实现 torch_cluster.radius：返回 (row, col)，row 递增。"""
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)
    d = torch.cdist(x, y)
    allowed = (batch_x.view(-1, 1) == batch_y.view(1, -1)) & (d <= r)
    rows, cols = [], []
    order = torch.argsort(batch_x)
    # 按 batch 分组逐行取邻居，等价按行号升序（同一 batch 内）
    for i in order.tolist():
        idx = torch.nonzero(allowed[i], as_tuple=False).view(-1)
        if idx.numel() > max_num_neighbors:
            idx = idx[:max_num_neighbors]
        rows.append(torch.full_like(idx, i))
        cols.append(idx)
    row = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long)
    col = torch.cat(cols) if cols else torch.zeros(0, dtype=torch.long)
    return row, col


def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        return edge_index, edge_attr[mask]
    return edge_index, None


def global_mean_pool(x, batch, size=None):
    n = int(batch.max()) + 1 if size is None else size
    out = torch.zeros((n, x.size(-1)), dtype=x.dtype, device=x.device)
    cnt = torch.zeros((n, 1), dtype=x.dtype, device=x.device)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones_like(x[:, :1]))
    return out / cnt.clamp(min=1)


def global_max_pool(x, batch, size=None):
    n = int(batch.max()) + 1 if size is None else size
    out = x.new_full((n, x.size(-1)), -float('inf'))
    for i in range(n):
        sel = x[batch == i]
        if sel.numel():
            out[i] = sel.max(dim=0).values
    return out


class _Storage:
    def __init__(self, row, col, value):
        self._row, self._col, self._value = row, col, value

    def row(self):
        return self._row

    def col(self):
        return self._col

    def value(self):
        return self._value


class SparseTensor:
    """只支持 get_geometry_indices 用到的子集：按行取值 / set_value / sum(dim=1) / storage。"""

    def __init__(self, row, col, value=None, sparse_sizes=None):
        self._row = row
        self._col = col
        self._value = value
        self._n_rows = int(sparse_sizes[0])
        counts = torch.bincount(row, minlength=self._n_rows)
        perm = torch.argsort(row, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        self._counts = counts
        self._starts = starts
        self._perm = perm

    def set_value(self, value):
        return SparseTensor(self._row, self._col, value, (self._n_rows, self._n_rows))

    def sum(self, dim=1):
        assert dim == 1
        return self._counts.to(torch.get_default_dtype())

    def __getitem__(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long).view(-1)
        pieces = []
        for r in idx.tolist():
            s = int(self._starts[r])
            pieces.append(self._perm[s:s + int(self._counts[r])])
        flat = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.long)
        new_row = torch.repeat_interleave(
            torch.arange(idx.numel()), torch.tensor([len(p) for p in pieces]))
        value = None if self._value is None else self._value[flat]
        return SparseTensor(new_row, self._col[flat], value, (idx.numel(), self._n_rows))

    @property
    def storage(self):
        return _Storage(self._row, self._col, self._value)


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add', **kwargs):
        super().__init__()
        self.aggr = aggr

    def reset_parameters(self):
        # torch_geometric 的基类提供空实现，GlobalMessagePassing 只重写了 init_parameters
        pass


def install():
    """把替身模块注册进 sys.modules，之后 import 原模型代码即可。"""
    tc = types.ModuleType('torch_cluster')
    tc.radius = radius
    sys.modules['torch_cluster'] = tc

    ts = types.ModuleType('torch_scatter')
    ts.scatter = scatter
    sys.modules['torch_scatter'] = ts

    tsp = types.ModuleType('torch_sparse')
    tsp.SparseTensor = SparseTensor
    sys.modules['torch_sparse'] = tsp

    tg = types.ModuleType('torch_geometric')
    tg_utils = types.ModuleType('torch_geometric.utils')
    tg_utils.remove_self_loops = remove_self_loops
    tg_nn = types.ModuleType('torch_geometric.nn')
    tg_nn.MessagePassing = MessagePassing
    tg_nn.global_mean_pool = global_mean_pool
    tg_nn.global_max_pool = global_max_pool
    tg.utils = tg_utils
    tg.nn = tg_nn
    sys.modules['torch_geometric'] = tg
    sys.modules['torch_geometric.utils'] = tg_utils
    sys.modules['torch_geometric.nn'] = tg_nn
