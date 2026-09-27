
    

import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_cluster import radius
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import global_max_pool, global_mean_pool 

from utils.utils import MLP
from utils.special import compute_geometry_features, BesselBasisLayer, SphericalBasisLayer, torsionLayer  
from layer.geom import LocalMessagePassing, GlobalMessagePassing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)
              
""" 自适应融合局部和全局特征 """

class AdaptiveMixtureLayer(nn.Module):
    def __init__(self, hidden_channels, adaptive_type='learned'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.adaptive_type = adaptive_type

        if adaptive_type == 'learned':
            self.ratio = nn.Parameter(torch.ones(2))

        elif adaptive_type == 'attention':
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 2)
            )

        elif adaptive_type == 'gate':
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),  
                nn.Linear(hidden_channels, hidden_channels),
                nn.Sigmoid()
            )

        self.local_trans = nn.Linear(hidden_channels, hidden_channels)
        self.global_trans = nn.Linear(hidden_channels, hidden_channels)

        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_local, x_global):
        x_local_trans = self.local_trans(x_local)
        x_global_trans = self.global_trans(x_global)

        if self.adaptive_type == 'learned':
            weights = F.softmax(self.ratio, dim=0)
            fused = weights[0] * x_local_trans + weights[1] * x_global_trans

        elif self.adaptive_type == 'attention':
            combined = torch.cat([x_local_trans, x_global_trans], dim=-1)
            weights = F.softmax(self.attention_net(combined), dim=-1)
            fused = weights[..., 0:1] * x_local_trans + weights[..., 1:2] * x_global_trans

        elif self.adaptive_type == 'gate':
            combined = torch.cat([x_local_trans, x_global_trans], dim=-1)
            gate = self.gate_net(combined)
            fused = gate * x_local_trans + (1 - gate) * x_global_trans
        else:
            fused = x_local_trans + x_global_trans

        fused = self.dropout(fused)
        fused = self.layer_norm(fused + x_local + x_global)
        return fused

class GRMP(nn.Module):
    def __init__(self, hidden_channels=128, g_cutoff=7.0, cutoff=5.0, num_spherical=3, num_radial=6, envelope_exponent=5, basis_emb_size_dist=128, 
                 basis_emb_size_angle=256, basis_emb_size_torsion=512, emb_channels=256, act=swish, pass_layers=3, adaptive_type='learned'):
        
        super().__init__()
        self.cutoff = cutoff
        self.g_cutoff=g_cutoff
        self.geom_layers = pass_layers
        self.mpnn_layers = pass_layers
        self.hidden_channels = hidden_channels
        self.node_embedding = nn.Linear(49, hidden_channels)
        # self.node_embedding = nn.Embedding(20, hidden_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.act = act

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)                              # [num_edges,6]
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)            # [num_angles,18]
        self.tbf=torsionLayer(num_spherical, num_radial, cutoff, envelope_exponent)                     # [num_torsions,54]

        self.mlp_global_rbf = MLP(num_radial, 64, basis_emb_size_dist, bias=False)
    
        self.message_layers = nn.ModuleList([
                    LocalMessagePassing(num_spherical, num_radial, hidden_channels, basis_emb_size_dist, basis_emb_size_angle, 
                                        basis_emb_size_torsion, act, emb_channels)  
                                            
            for _ in range(self.geom_layers)])
        
        self.global_layers = nn.ModuleList([GlobalMessagePassing(hidden_channels, pass_layers, g_cutoff, act) 
            for _ in range(self.geom_layers)])
        
        self.adaptive_layers = nn.ModuleList([
                AdaptiveMixtureLayer(hidden_channels, adaptive_type=adaptive_type) for _ in range(self.geom_layers)])

        self.layer_att = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1) 
        )

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 3, 512), 
            # torch.nn.Linear(hidden_channels * 4, 512), 
            torch.nn.ReLU(inplace=True),     
            torch.nn.Linear(512, 256), 
            torch.nn.ReLU(inplace=True),   
            torch.nn.Linear(256, 128),       
            torch.nn.ReLU(inplace=True),   
            torch.nn.Linear(128, 64),        
            torch.nn.ReLU(inplace=True),   
            torch.nn.Linear(64, 1)           
        )
   
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.global_layers:
            layer.reset_parameters()
        self.node_embedding.reset_parameters()
    
    def get_edge_info(self, edge_index, pos):
        edge_index, _ = remove_self_loops(edge_index)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, dist 

    def forward(self, data):
        x_geom, pos, edge_index, batch = data.x_geom, data.pos, data.edge_index,data.batch

        row, col = radius(pos, pos, self.g_cutoff, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, dist_g = self.get_edge_info(edge_index_g, pos)    
        results = compute_geometry_features(pos, edge_index, batch)
        dists = results['dists']
        angles = results['angles']
        torsions = results['torsions']

        if len(angles) == 0:
            # 没有角度时
            angles = torch.zeros(0, device=pos.device)
    
        if len(torsions) == 0:
            # 没有二面角时
            torsions = torch.zeros(0, device=pos.device)
            results['torsion_to_angle_map'] = torch.zeros(0, device=pos.device, dtype=torch.long)
            
        j, i = results['edge_index'][0], results['edge_index'][1]
        idx_kj = results['idx_kj']
        idx_ji = results['idx_ji']
        idx_kj_t = results['idx_kj_t']
        idx_ji_t = results['idx_ji_t']
        torsion_to_angle_map = results['torsion_to_angle_map']

        rbf = self.rbf(dists)    
        sbf = self.sbf(dists, angles, idx_kj)    
        tbf = self.tbf(dists, angles, torsions, idx_kj_t, torsion_to_angle_map)

        x_node = self.node_embedding(x_geom)     

        # 不在图上的边            
        rbf_g = self.rbf(dist_g)
        edge_attr_rbf_g = self.mlp_global_rbf(rbf_g)

        layer_geom_feas = []
        x_current = x_node

        for layer in range(self.geom_layers):
            x_local = self.message_layers[layer](x_current, rbf, j, i, sbf, idx_kj, tbf, torsion_to_angle_map, idx_ji_t, idx_ji)
            x_global = self.global_layers[layer](x_current, edge_attr_rbf_g, edge_index_g)
            x_current = self.adaptive_layers[layer](x_local, x_global)
            layer_geom_feas.append(x_current) 

        geom = torch.cat(layer_geom_feas, dim=-1)  # [N, L*H] 
        # print("geom shape:",geom.shape)                

        xt_geom = global_mean_pool(geom, batch)
        # print("xt_geom shape:",xt_geom.shape)   

        out = self.MLP(xt_geom)
        
        return out 