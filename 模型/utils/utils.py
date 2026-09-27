import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)

class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class MLP(nn.Sequential):
    def __init__(self, *channels, bias=True):
        if len(channels) == 1 and isinstance(channels[0], (list, tuple)):
            channels = channels[0]
        else:
            channels = list(channels)
        
        layers = []
        for i in range(1, len(channels)):
            layers.append(Linear(channels[i-1], channels[i], bias=bias))
            layers.append(SiLU())
        
        super().__init__(*layers)
    
    def reset_parameters(self):

        for module in self:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor)
        scale /= ((tensor.size(-1) + tensor.size(-2)) * tensor.var())
        tensor.data *= scale.sqrt()
        
class BaseMessagePassing(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin = MLP([hidden_channels, hidden_channels])  
    
    def reset_parameters(self):
        self.lin.reset_parameters()