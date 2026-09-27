import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from utils.utils import MLP, ResidualLayer, glorot_orthogonal

class LocalMessagePassing(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, hidden_channels, basis_emb_size_dist, basis_emb_size_angle, 
                 basis_emb_size_torsion, act, emb_channels):

        super(LocalMessagePassing, self).__init__()
        self.act = act
        self.emb_channels = emb_channels        
        self.hidden_dim = hidden_channels
        
        self.mlp_x1 = MLP([self.hidden_dim, self.hidden_dim])
        
        self.mlp_m_ji = MLP([3 * self.hidden_dim, self.hidden_dim])          # 直接边消息
        self.mlp_m_kj = MLP([3 * self.hidden_dim, self.hidden_dim])          # 邻居边消息
               
        # 几何特征
        self.mlp_sbf = MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim])     
        
        # rbf 特征
        self.mlp_rbf = MLP(num_radial, 64, basis_emb_size_dist, bias=False)
        self.lin_rbf = nn.Linear(basis_emb_size_dist, self.hidden_dim, bias=False)  

        # sbf 特征
        self.mlp_sbf1 = MLP(num_spherical * num_radial, 128, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, self.hidden_dim, bias=False)
        
        # tbf 特征
        self.mlp_tbf1 = MLP(num_spherical * num_spherical * num_radial, 256, basis_emb_size_torsion, bias=False)
        self.lin_tbf2 = nn.Linear(basis_emb_size_torsion, self.hidden_dim, bias=False)
        
        # 扭转角映射
        self.mlp_angle_for_torsion = MLP([self.hidden_dim, self.hidden_dim])

        self.lin_rbf_out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.mlp_x2 = MLP([self.hidden_dim, self.hidden_dim])

        self.res_layers = nn.ModuleList([
            ResidualLayer(hidden_channels) for _ in range(3)  
        ]) 

        self.reset_parameters()
    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf_out.weight, scale=2.0)  
        self.mlp_x2.reset_parameters()  

    def forward(self, x, rbf, j, i, sbf, idx_kj, tbf, torsion_to_angle_map, idx_ji_t, idx_ji): 
        res_x = x
        x = self.mlp_x1(x)
        
        # rbf
        rbf_f = self.mlp_rbf(rbf)        
        rbf_feat= self.lin_rbf(rbf_f)

        m = torch.cat([x[i], x[j], rbf_f], dim=-1)               
        m_ji = self.act(self.mlp_m_ji(m))                                           
        m_kj = self.act(self.mlp_m_kj(m))                        

        m_neighbor = m_kj * self.lin_rbf(rbf_feat)

        # sbf
        sbf_feat = self.mlp_sbf1(sbf)
        sbf_feat = self.lin_sbf2(sbf_feat)        
        m_angle = m_neighbor[idx_kj] * self.mlp_sbf(sbf_feat)

        # tbf       
        tbf_feat = self.mlp_tbf1(tbf)
        tbf_feat = self.lin_tbf2(tbf_feat)
        m_angle_mapped = m_angle[torsion_to_angle_map]
        m_angle_processed = self.mlp_angle_for_torsion(m_angle_mapped)

        m_torsion = m_angle_processed * tbf_feat     
        m_torsion_agg = scatter(m_torsion, idx_ji_t, dim=0,dim_size=m.size(0), reduce='add')

        m = m_ji + m_torsion_agg                                                      

        m_final = self.lin_rbf_out(rbf_feat) * m                         # 距离信息直接地影响最终消息，调制阶段：作为门控信号，控制整合后消息的强度       
    
        x = x + scatter(m_final, i, dim=0, dim_size=x.size(0), reduce='add')
        # print("first x shape:",x.shape)                                # [nodes,hidden]

        x = self.mlp_x2(x)  
        # print("after mlp_x2 x shape:",x.shape)                         #[nodes, 2 * hidden]
        if len(self.res_layers) > 0:
            x = self.res_layers[0](x) + res_x
            for res_layer in self.res_layers[1:]:
                x = res_layer(x) 
        return x

class GlobalMessagePassing(MessagePassing):
    def __init__(self, hidden_channels, num_layers, g_cutoff, act):
        super(GlobalMessagePassing, self).__init__(aggr='add')

        self.hidden_dim = hidden_channels
        self.g_layers = num_layers
        self.g_cutoff = g_cutoff
        self.act = act 
      
        # 节点特征
        self.mlp_x1 = MLP([self.hidden_dim, self.hidden_dim])
        self.mlp_x2 = MLP([self.hidden_dim, self.hidden_dim])
        

        self.mlp_m = MLP([self.hidden_dim * 3, self.hidden_dim])
        self.W_edge_attr = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.mlp_edge_update = MLP([self.hidden_dim * 2, self.hidden_dim, self.hidden_dim])        
    
        self.res_layers = nn.ModuleList([
            ResidualLayer(hidden_channels) for _ in range(3)  
        ])         
        self.init_parameters()
    
    def init_parameters(self):
        glorot_orthogonal(self.W_edge_attr.weight, scale=2.0)
        self.mlp_edge_update.reset_parameters()
    
    def forward(self, x, edge_attr, edge_index):
        res_x = x
        x = self.mlp_x1(x)        
        # 边特征更新
        j, i = edge_index  
        edge_feats = self.mlp_edge_update(torch.cat([x[i], x[j]], dim=-1))
        com_edge = edge_attr + edge_feats  
    
        m = torch.cat([x[i], x[j], com_edge], dim=-1)
        m = self.mlp_m(m)
        message = m * self.W_edge_attr(com_edge)
        
        # 聚合到目标节点
        message_agg = scatter(message, i, dim=0, dim_size=x.size(0), reduce='sum')
        
        # 节点更新
        x = x + message_agg  
        x = self.mlp_x2(x)

        if len(self.res_layers) > 0:
            x = self.res_layers[0](x) + res_x
            for res_layer in self.res_layers[1:]:
                x = res_layer(x)       
        return x
    
    def message(self, x_i, x_j, com_edge):

        m = torch.cat([x_i, x_j, com_edge], dim=-1)
        m = self.mlp_m(m)
        message = m * self.W_edge_attr(com_edge)
        return message
    
    def aggregate(self, messages, edge_index):

        _, i = edge_index
        return scatter(messages, i, dim=0, reduce='sum')
    
    def update(self, x, aggr_out):

        x = x + aggr_out
        x = self.mlp_x2(x)
        
        return x