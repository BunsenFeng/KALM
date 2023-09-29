import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MessagePassing, RGCNConv, GATConv
from torch_geometric.utils import softmax
import torch.nn.functional as F

class SimpleHGN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim, rel_emb, beta=None, final_layer=False):
        super(SimpleHGN, self).__init__(aggr = "add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3*out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding.from_pretrained(rel_emb)
        #self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        #self.rel_emb.from_pretrained(rel_emb)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                    
    def forward(self, x, edge_index, edge_type, pre_alpha=None):
        
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        if self.final:
            output = F.normalize(output, dim=1)
            
        return output, self.alpha.detach()
      
    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)
        return out

    def update(self, aggr_out):
        return aggr_out

# rel_emb = torch.randn(100000, 100)
# model = SimpleHGN(in_channels = 100, out_channels = 100, num_edge_type = 100000, rel_dim = 100, rel_emb = rel_emb)
# node_features = torch.randn(20,100)
# edge_index = torch.tensor([[1,3,5],[2,4,6]]).long()
# edge_type = torch.tensor([456,11234,2]).long()
# print(model(node_features, edge_index, edge_type)[0].size())

# gated RGCN in a nutshell
# in_channels: text encoding dimension, out_channels: dim for each node rep, num_relations
# Input: node_features:torch.size([node_cnt,in_channels]), query_features = torch.size([in_channels]) (MISSING IN DATA FILE)
# edge_index = torch.size([[headlist],[taillist]]), edge_type = torch.size([typelist])
# Output: node representation of torch.size([node_cnt, out_channels])
class GatedRGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(GatedRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.RGCN1 = RGCNConv(in_channels = out_channels, out_channels = out_channels, num_relations = num_relations)
        self.attention_layer = nn.Linear(2 * out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        nn.init.xavier_uniform_(self.attention_layer.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, node_features, edge_index, edge_type):

        #layer 1
        #print(node_features.size())
        #print(edge_index.size())
        #print(edge_type.size())
        u_0 = self.RGCN1(node_features, edge_index, edge_type)
        a_1 = self.sigmoid(self.attention_layer(torch.cat((u_0, node_features),dim=1)))
        h_1 = self.tanh(u_0) * a_1 + node_features * (1 - a_1)

        return h_1