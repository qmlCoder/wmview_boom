import math
import torch
import numpy as np
import sympy as sym
from scipy.special import jv
from scipy.optimize import brentq
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sph_harm_prefactor(k, m):
    return ((2 * k + 1) * math.factorial(k - abs(m)) /
            (4 * math.pi * math.factorial(k + abs(m))))**0.5

def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * jv(n + 0.5, r)

def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i, ))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    return zerosj

def spherical_bessel_formulas(n):
    x = sym.symbols('x')
    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x)**i)]
        a = sym.simplify(b)
    return f

def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]
    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis

def associated_legendre_polynomials(k, zero_m_only=True):
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] -
                                        (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))
    return P_l_m

def real_sph_harm(k, zero_m_only=True, spherical_coordinates=True):
    if not zero_m_only:
        # 使用符号而不是数值初始化
        x, y = sym.symbols('x y')
        S_m = [sym.sympify(0)]  # 使用sympify确保是符号
        C_m = [sym.sympify(1)]  # 使用sympify确保是符号
        for i in range(1, k):
            S_m.append(x * S_m[i - 1] + y * C_m[i - 1])
            C_m.append(x * C_m[i - 1] - y * S_m[i - 1])

    P_l_m = associated_legendre_polynomials(k, zero_m_only)
    
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z_sym = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if isinstance(P_l_m[i][j], sym.Expr):  # 只对符号表达式进行替换
                    P_l_m[i][j] = P_l_m[i][j].subs(z_sym, sym.cos(theta))
        
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                if isinstance(S_m[i], sym.Expr) and S_m[i] != 0:  # 只对非零符号表达式进行替换
                    S_m[i] = S_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(
                                         y, sym.sin(theta) * sym.sin(phi))
            for i in range(len(C_m)):
                if isinstance(C_m[i], sym.Expr) and C_m[i] != 0:  # 只对非零符号表达式进行替换
                    C_m[i] = C_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(
                                         y, sym.sin(theta) * sym.sin(phi))

    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(k)]
    
    for i in range(k):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m

class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        env_val = 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p1 * x
        zero = torch.zeros_like(x)
        return torch.where(x < 1, env_val, zero)

class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=6):
        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = torch.nn.Parameter(torch.empty(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(math.pi)
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()

class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf
        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out

class torsionLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                envelope_exponent=5):
        super(torsionLayer, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: torch.zeros_like(x) + torch.zeros_like(y) + sph1(0,0)) #torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k+i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, torsion, idx_kj, angle_idx_map):
        """
        dist: [num_edges] 距离
        angle: [num_angles] 角度值  
        torsion: [num_torsions] 二面角值
        idx_kj: [num_torsions] 二面角对应的k-j边索引
        angle_idx_map: [num_torsions] 二面角到角度的索引映射
        """
        dist = dist / self.cutoff
        
        # 计算径向基函数
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        
        # 使用映射找到每个二面角对应的角度值
        angle_for_torsion = angle[angle_idx_map]  # [num_torsions]
        
        # 计算球谐基函数
        cbf = torch.stack([f(angle_for_torsion, torsion) for f in self.sph_funcs], dim=1)
        
        # 选择对应的rbf特征
        rbf_selected = rbf[idx_kj]
        
        # 计算张量积
        n, k = self.num_spherical, self.num_radial
        rbf_reshaped = rbf_selected.view(-1, n, k)
        cbf_reshaped = cbf.view(-1, n, n)
        
        out = (rbf_reshaped.unsqueeze(1) * cbf_reshaped.unsqueeze(-1)).view(-1, n * n * k)
        return out

class torsion_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: torch.zeros_like(x) + torch.zeros_like(y) + sph1(0,0)) #torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k+i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        return out    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_geometry_indices(edge_index: torch.Tensor, num_nodes: int):
    """
    返回:
        triplet:  idx_i, idx_j, idx_k, idx_kj, idx_ji
        quad:     idx_i_t, idx_j_t, idx_k_t, idx_l,
                  idx_kj_t, idx_ji_t, idx_kl_t
        angle_idx_map:  quad -> triplet 的一一对应索引
    """
    row, col = edge_index  # j->i
    N = row.size(0)
    value = torch.arange(N, device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))

    # i-j-k
    adj_t_row = adj_t[row]               # 对每条边 (j->i) 取 j 的邻居 k
    num_triplets = adj_t_row.set_value(None).sum(dim=1).long()  # 每条边的三元组数

    idx_i = col.repeat_interleave(num_triplets)  # 中心节点 i
    idx_j = row.repeat_interleave(num_triplets)  # 中心节点 j
    idx_k = adj_t_row.storage.col()              # 邻居 k
    edge_kj = adj_t_row.storage.value()          # 原始图中 k->j 的边编号

    # 去掉 i=k 
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj = edge_kj[mask]

    idx_ji = torch.arange(N, device=row.device).repeat_interleave(num_triplets)[mask]

    # i-j-k-l
    adj_t_k = adj_t[idx_k]                       
    num_quads = adj_t_k.set_value(None).sum(dim=1).long()

    idx_i_t = idx_i.repeat_interleave(num_quads)
    idx_j_t = idx_j.repeat_interleave(num_quads)
    idx_k_t = idx_k.repeat_interleave(num_quads)
    idx_l   = adj_t_k.storage.col()
    edge_kl = adj_t_k.storage.value()

    # 过滤：l != i 且 l != j
    mask_q = (idx_l != idx_i_t) & (idx_l != idx_j_t)
    idx_i_t, idx_j_t, idx_k_t, idx_l = idx_i_t[mask_q], idx_j_t[mask_q], idx_k_t[mask_q], idx_l[mask_q]
    idx_kj_t = idx_kj.repeat_interleave(num_quads)[mask_q]
    idx_ji_t = idx_ji.repeat_interleave(num_quads)[mask_q]
    idx_kl_t = edge_kl[mask_q]


    angle_keys = torch.stack([idx_i, idx_j, idx_k], dim=1)          # [M,3]
    quad_keys  = torch.stack([idx_i_t, idx_j_t, idx_k_t], dim=1)    # [Q,3]

    match = (quad_keys.unsqueeze(1) == angle_keys.unsqueeze(0)).all(dim=2)
    angle_idx_map = match.max(dim=1)[1]        # 每行第一个 True 的下标即为对应 angle 索引

    return (idx_i, idx_j, idx_k, idx_kj, idx_ji,
            idx_i_t, idx_j_t, idx_k_t, idx_l,
            idx_kj_t, idx_ji_t, idx_kl_t, angle_idx_map)

def compute_geometry_features(pos: torch.Tensor,
                              edge_index: torch.Tensor,
                              batch=None):
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    edge_index, _ = remove_self_loops(edge_index)
    j, i = edge_index  # j->i
    dist = (pos[i] - pos[j]).norm(dim=1)

    (idx_i, idx_j, idx_k, idx_kj, idx_ji,
     idx_i_t, idx_j_t, idx_k_t, idx_l,
     idx_kj_t, idx_ji_t, idx_kl_t, angle_idx_map) = \
        get_geometry_indices(edge_index, num_nodes=pos.size(0))

    # 角度
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    angle = torch.atan2(torch.cross(pos_ji, pos_jk).norm(dim=1),
                        (pos_ji * pos_jk).sum(dim=1))

    # 二面角
    if len(idx_i_t) > 0:
        pos_ij = pos[idx_j_t] - pos[idx_i_t]
        pos_jk = pos[idx_k_t] - pos[idx_j_t]
        pos_kl = pos[idx_l]   - pos[idx_k_t]

        n1 = torch.cross(pos_ij, pos_jk)
        n2 = torch.cross(pos_jk, pos_kl)

        b = (torch.cross(n1, n2) * pos_jk).sum(dim=1) / pos_jk.norm(dim=1)
        torsion = torch.atan2(b, (n1 * n2).sum(dim=1))
        # 映射到对应角度
        torsion_angle = angle[angle_idx_map]
    else:
        torsion = torch.empty(0, dtype=pos.dtype, device=pos.device)
        torsion_angle = torch.empty(0, dtype=pos.dtype, device=pos.device)
        angle_idx_map = torch.empty(0, dtype=torch.long, device=pos.device)

    return {
        'dists': dist,
        'angles': angle,
        'torsions': torsion,
        'torsion_angles': torsion_angle,   # 与 torsion 一一对应的角度
        'angle_indices': (idx_i, idx_j, idx_k),
        'torsion_indices': (idx_i_t, idx_j_t, idx_k_t, idx_l),
        'edge_index': edge_index,
        'idx_kj': idx_kj,
        'idx_ji': idx_ji,
        'idx_kj_t': idx_kj_t,
        'idx_ji_t': idx_ji_t,
        'idx_kl_t': idx_kl_t,
        'torsion_to_angle_map': angle_idx_map
    }