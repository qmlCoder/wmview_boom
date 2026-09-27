import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from sklearn.preprocessing import OneHotEncoder
from mendeleev import element
from pymatgen.core.structure import IMolecule
from torch_geometric.data import Data, Dataset

def get_hybridization_one_hot(atom):
    """获取杂化类型 one-hot 编码"""
    atom_type = atom.GetSymbol()
    
    # H、卤素原子没有杂化类型，编码为 none
    if atom_type in ['H', 'Cl', 'Br', 'I']:
        return [0, 0, 0, 0, 1]  # none
    
    hybridization = atom.GetHybridization()
    
    # 杂化类型映射
    hybridization_map = {
        rdchem.HybridizationType.SP: [1, 0, 0, 0, 0],    # sp
        rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0],   # sp2
        rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0],   # sp3
        rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0],  # sp3d
    }
    
    return hybridization_map.get(hybridization, [0, 0, 0, 0, 1])  

FEATURES = {
    'atom_symbol': ['H', 'C', 'N', 'O'],  
}

# 总维度：4 + 1 + 1 + 1 + 5 + 1 + 1 + 1 + 1 = 16维
atom_dim = 16

def atom_features(atom, mol) -> list:

    feats = []
    atom_type = atom.GetSymbol()
    # 1. 元素类型 atom_type (4 one-hot)
    feats.extend(one_hot(atom_type, FEATURES['atom_symbol']))

    # 2. 度 degree (integer)
    degree = float(atom.GetDegree())
    feats.append(degree)
    
    # 3. 形式电荷 formal_charge (integer)
    formal_charge = float(atom.GetFormalCharge())
    feats.append(formal_charge)
    
    # 4. 氢原子数 num_hydrogens (integer)
    num_hs = float(atom.GetTotalNumHs())
    feats.append(num_hs)

    # 5. 杂化类型 hybridization (5 one-hot)
    hybrid_val = get_hybridization_one_hot(atom)
    feats.extend(hybrid_val)
    
    # 6. 芳香性 aromatic (one-hot)
    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    feats.append(is_aromatic)
    
    # 7. 环的数量 ring_num (integer)
    ring_info = mol.GetRingInfo()
    atom_ring_num = 0
    if ring_info.NumAtomRings(atom.GetIdx()) > 0:
        atom_ring_num = float(ring_info.NumAtomRings(atom.GetIdx()))
    feats.append(atom_ring_num)

    # 8. 隐式价 implicit_valence (integer)
    implicit_valence = float(atom.GetImplicitValence())
    feats.append(implicit_valence)

    # 9. 原子质量 atomic_mass (integer)
    atomic_mass = float(atom.GetMass())
    feats.append(atomic_mass)

    if len(feats) != atom_dim:
        raise ValueError(f"Feature dimension error.")
    
    return feats

def one_hot(value, allowable_set):
    """One-hot编码"""
    encoding = [0] * len(allowable_set)
    for i, item in enumerate(allowable_set):
        if value == item:
            encoding[i] = 1
            break
    return encoding

def get_bond_type_one_hot(bond):
    """获取键类型的one-hot编码"""
    bond_type = bond.GetBondType()
    
    # 定义键类型列表
    bond_types = [
        Chem.BondType.SINGLE,      # 单键
        Chem.BondType.DOUBLE,      # 双键
        Chem.BondType.TRIPLE,      # 三键
        Chem.BondType.AROMATIC,    # 芳香键
    ]
    
    # 检查是否是已知类型
    if bond_type in bond_types:
        # 找到对应的索引
        for i, bt in enumerate(bond_types):
            if bond_type == bt:
                encoding = [0] * len(bond_types)
                encoding[i] = 1
                return encoding
    
    # 如果不在列表中，返回全0
    return [0] * len(bond_types)

def get_bond_is_conjugated(bond):
    """获取键是否共轭- 键的属性"""
    return 1.0 if bond.GetIsConjugated() else 0.0

def get_bond_in_ring(bond):
    """获取键是否在环中- 计算键本身是否在环上"""
    return 1.0 if bond.IsInRing() else 0.0

BOND_FEATURES = {
    # 'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE'],  # 3维 one-hot
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
}

# 边特征总维度：4 + 1 + 1 = 6维
BOND_DIM = 6

def bond_features(bond) -> list:
    """提取边（键）特征
    
    Args:
        bond: RDKit键对象
        mol: RDKit分子对象
        
    Returns:
        list: 边特征向量 (7维)
    """
    feats = []
    
    # 1. 键类型 (4维 one-hot) 
    bond_type_encoding = get_bond_type_one_hot(bond)
    feats.extend(bond_type_encoding)
    
    # 2. 是否共轭 (one-hot)
    conjugated_val = get_bond_is_conjugated(bond)
    feats.append(conjugated_val)
    
    # 3. 键是否在环中 计算键本身是否在环上，不同于原子的IsInRing(), (one-hot)
    in_ring_val = get_bond_in_ring(bond)
    feats.append(in_ring_val)

    if len(feats) != BOND_DIM:
        raise ValueError(f"边特征维度错误: 期望{BOND_DIM}, 实际{len(feats)}")
    
    return feats

atomic_number_map = {
    'H': 1,    # 氢
    'He': 2,   # 氦
    'Li': 3,   # 锂
    'Be': 4,   # 铍
    'B': 5,    # 硼
    'C': 6,    # 碳
    'N': 7,    # 氮
    'O': 8,    # 氧
    'F': 9,    # 氟
    'Ne': 10,  # 氖
    'Na': 11,  # 钠
    'Mg': 12,  # 镁
    'Al': 13,  # 铝
    'Si': 14,  # 硅
    'P': 15,   # 磷
    'S': 16,   # 硫
    'Cl': 17,  # 氯
    'Ar': 18,  # 氩
    'K': 19,   # 钾
    'Ca': 20,  # 钙
}

def read_xyz_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_atoms = int(lines[0].strip())
    R = []
    Z = []
    
    for line in lines[2:num_atoms + 2]:
        parts = line.split()
        if len(parts) >= 4:
            atom_type = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            R.append([x, y, z])
            atomic_number = atomic_number_map.get(atom_type, None)
            if atomic_number is not None:
                Z.append(atomic_number)
            else:
                print(f"Warning: Unknown atomic type '{atom_type}' found.")
    
    R_array = np.array(R)
    Z_array = np.array(Z)
    return R_array, Z_array

def atom():
    feature_lists = {
        'number': [],
        'ionenergies': [],
        'atomic_radius': [],
        # 'group_id': [],     
        'period_list': [],  
        'nvalence': [],     
        # 'en_pauling': [],   
        'atomic_volume': [], 
        'electron_affinity': []   
    }
    ele_nums = [1] + list(range(3, 10)) 
    for i in ele_nums:
        Ele = element(i)
        feature_lists['number'].append(i)                            
        feature_lists['ionenergies'].append(Ele.ionenergies[1])      
        feature_lists['atomic_radius'].append(Ele.atomic_radius)
        # feature_lists['group_id'].append(Ele.group_id)
        feature_lists['period_list'].append(Ele.period)
        feature_lists['nvalence'].append(Ele.nvalence())   
        # feature_lists['en_pauling'].append(Ele.en_pauling)
        feature_lists['atomic_volume'].append(Ele.atomic_volume)
        feature_lists['electron_affinity'].append(Ele.electron_affinity)  

    encoded_features = {}             
    for feature_name, feature_values in feature_lists.items():  

        # print(f"{feature_name}: {len(set(feature_values))} unique values")  
        #          
        feature_array = np.array(feature_values).reshape(-1, 1)
        OHE = OneHotEncoder()         
        OHE.fit(feature_array)       
        encoded_features[feature_name] = OHE.transform(feature_array).toarray()  

    return tuple(encoded_features.values())     

a = atom()

def get(mol):
    all_nbrs = []
    atom_fea_list = []

    for atom in mol:
        nbrs = mol.get_neighbors(atom, radius)
        all_nbrs.append(nbrs)
        number = atom.specie.number
        n = number - 1
        arrays_to_concatenate = [arr[n] for arr in a]
        result = np.concatenate(arrays_to_concatenate, axis=0)
        atom_fea_list.append(result)

    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] 
    atom_fea = torch.Tensor(np.array(atom_fea_list))   

    return all_nbrs, atom_fea  

def format_adj_matrix(adj_matrix):
    size = len(adj_matrix)  
    src_list = list(range(size))  
    
    all_src_nodes = torch.tensor([[x] * adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
    all_dst_nodes = adj_matrix.view(-1).unsqueeze(0) 

    return torch.cat((all_src_nodes, all_dst_nodes), dim=0)

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax  
        assert dmax - dmin > step  
        self.filter = np.arange(dmin, dmax + step, step)  
        if var is None:
            var = step  
        self.var = var  

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                       self.var ** 2)  
radius = 8
max_num_nbr = 12
gdf = GaussianDistance(0, radius, 0.2)

class MyDataset(Dataset):
    def __init__(self, target_file, structure_dir):
        super(MyDataset, self).__init__()
        target_df = pd.read_csv(target_file, header=0)   
        self.structure_dir = structure_dir  
        self.filename = target_df.loc[:, 'filename'].values  
        self.smiles = target_df.loc[:, 'smiles'].values  

        self.target_M = target_df.loc[:, 'density(g/cm3)'].values 
        self.target_F = target_df.loc[:, 'solid_HOF(kcal/mol)'].values 
        self.target_Q = target_df.loc[:, 'Q(cal/g)'].values  
        self.target_D = target_df.loc[:, 'D(Km/s)'].values  
        self.target_p = target_df.loc[:, 'p(GPa)'].values  

        # self.target_bde = target_df.loc[:, 'BDE'].values 

    def __getitem__(self, index):
        filename = self.filename[index]    
        smiles = self.smiles[index]  
        M_target = self.target_M[index]  
        F_target = self.target_F[index]
        Q_target = self.target_Q[index]
        D_target = self.target_D[index]
        p_target = self.target_p[index] 

        # bde_target = self.target_bde[index] 

        mol = IMolecule.from_file(self.structure_dir +'/'+ str(filename))  
        MOL = Chem.MolFromSmiles(smiles)  
        MOL = Chem.AddHs(MOL)  

        # 提取二维特征
        x_2d = torch.tensor([atom_features(a,MOL) for a in MOL.GetAtoms()], dtype=torch.float)

        edge_index, edge_attr = [], []
        for bond in MOL.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = bond_features(bond)

            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([feat, feat])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 获取邻居和原子特征
        all_nbrs, atom_fea = get(mol)
        nbr_fea_idx, nbr_fea = [], []

        # 处理几何neighbor特征
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = gdf.expand(nbr_fea)  
        nbr_fea = torch.Tensor(nbr_fea).view(-1, 41)        
        nbr_fea_idx = format_adj_matrix(torch.LongTensor(nbr_fea_idx))

        # 提取三维特征
        file_path = os.path.join(self.structure_dir, filename)
        R_i, z_i = read_xyz_file(file_path)
        pos = torch.tensor(R_i, dtype=torch.float)
        z = torch.tensor(z_i, dtype=torch.long)

        if len(z) != MOL.GetNumAtoms():
            raise ValueError(f'Atom count mismatch for smiles {smiles} and XYZ file {filename}.')         

        M_target = torch.Tensor([float(M_target)])
        F_target = torch.Tensor([float(F_target)]) 
        Q_target = torch.Tensor([float(Q_target)])
        D_target = torch.Tensor([float(D_target)])
        p_target = torch.Tensor([float(p_target)]) 
        # bde_target = torch.Tensor([float(bde_target)])

# first_data:Data(edge_index=[2, 18], edge_attr=[18, 6], pos=[9, 3], x_graph=[9, 16], x_geom=[9, 64], z=[9], 
# geom_edge_index=[2, 108], geom_edge_attr=[108, 41], M=[1], F=[1], Q=[1], D=[1], p=[1], smiles='O=[N+]([O-])c1cnno1')

# all
        graph = Data(x_graph=x_2d, edge_index=edge_index, edge_attr=edge_attr,                     
                    x_geom=atom_fea, pos=pos, z=z, geom_edge_index=nbr_fea_idx, geom_edge_attr=nbr_fea,            
                    M=M_target, F=F_target, Q=Q_target, D=D_target, p=p_target, 
                    smiles=smiles)
# 2d
        # graph = Data(x_graph=x_2d, edge_index=edge_index, edge_attr=edge_attr,                               
        #             M=M_target, F=F_target, Q=Q_target, D=D_target, p=p_target, 
        #             smiles=smiles)

# geom      
        # graph = Data(x_geom=atom_fea, pos=pos, z=z, geom_edge_index=nbr_fea_idx, geom_edge_attr=nbr_fea,            
        #             M=M_target, F=F_target, Q=Q_target, D=D_target, p=p_target, 
        #             smiles=smiles)

        return graph 
     
    def __len__(self):
        return len(self.filename)

target_file = '/export/home/zhengym/model/fusion/FUSED-mpnn/data/new_data/1_final_123679.csv' 
structure_dir = '/export/home/zhengym/model/data/data_dft/gdb/gdb_xyz' 
dataset = MyDataset(target_file, structure_dir)
data_list = []
for item in dataset:
    data_list.append(item)
save_path = '/export/home/zhengym/model/fusion/FUSED-mpnn/data/new_data/pt/new_123679_49.pt'
torch.save(data_list, save_path)
print("Done!")
            