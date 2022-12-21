import dgl.function as fn
#
import torch
from torch import nn
import torch.nn.functional as F 
from torch_geometric.nn import (
    GCNConv, GINEConv, GINConv, TopKPooling, GATConv, GATv2Conv, JumpingKnowledge)
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
#import pytorch_lightning as pl

# custom model blocks 
from .model_utils import init_weights_xavier   # *

def get_model(dim_feats:int, out_dim:int, name:str="GIN"):
    model = GIN
    gnn_model = model(FeaturesArgs.n_node_features, 1)
    gnn_model.apply(init_weights_xavier)
    return gnn_model

class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            print(f"Feature shape:{h_total.shape}")
            return self.linear(h_total)


class GAT(torch.nn.Module):
    def __init__(self, dim_feats:int, num_class:int, dim_h:int=32, 
                 conv_dropout=0.6, lin_dropout=0.6, activation:str=None):
        super(GAT, self).__init__()
        self.in_head = 8
        self.out_head = 1
        self.lin_dropout = lin_dropout
        self.conv_dropout = conv_dropout
        self.activation = activation
        #
        self.conv1 = GATv2Conv(
            dim_feats, self.dim_h, heads=self.in_head, dropout=0.6)
        self.conv2 = GATv2Conv(
            dim_h*self.in_head, num_class, concat=False,
            heads=self.out_head, dropout=conv_dropout)

    def forward(self, x , edge_index):
        x = F.dropout(x, p=self.lin_dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.lin_dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.activation == "classification":
            return F.log_softmax(h, dim=1)
        return F.leaky_relu(h)
    
class GCN(nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)
        return out, hidden

class GIN(nn.Module):
    """GIN"""
    def __init__(self, dim_feats:int, n_class:int, activation:str=None, 
                 dim_h:int=32):
        super(GIN, self).__init__()
        self.activation = activation
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(dim_feats, dim_h),
                          nn.BatchNorm1d(dim_h), nn.LeakyReLU(),
                          nn.Linear(dim_h, dim_h), nn.LeakyReLU()))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h),
                          nn.BatchNorm1d(dim_h), nn.LeakyReLU(),
                          nn.Linear(dim_h, dim_h), nn.LeakyReLU()))
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h),
                          nn.BatchNorm1d(dim_h), nn.LeakyReLU(),
                          nn.Linear(dim_h, dim_h), nn.LeakyReLU()))
        
        self.lin1 = nn.Linear(dim_h*3, dim_h*3)
        self.lin2 = nn.Linear(dim_h*3, n_class)

    def forward(self, x, x_e, edge_index, batch):
        # 
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)  # (*, 32)
        h2 = global_add_pool(h2, batch)  # (*, 32)
        h3 = global_add_pool(h3, batch)  # (*, 32)
        
        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        if self.activation == "classification":
            return F.log_softmax(h, dim=1)
        return F.leaky_relu(h)
    
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(
                nn.Sequential(nn.Linear(input_dim, hidden_dim),    
                              nn.ReLU(), 
                              nn.Linear(hidden_dim, hidden_dim))
            )

    def forward(self, x, edge_index, batch):
        # if data.num_node_features == 0:
        #     x = torch.ones(data.num_nodes, 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, 
                          training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_add_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
# Jumping knowledge model
# model = nn.Sequential('x, edge_index, batch', [
#             (nn.Dropout(p=0.5), 'x -> x'),
#             (GCNConv(dim_feats, dim_h), 'x, edge_index -> x1'),
#             ReLU(inplace=True),
#             (GCNConv(dim_h, dim_h), 'x1, edge_index -> x2'),
#             ReLU(inplace=True),
#             (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
#             (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
#             (global_mean_pool, 'x, batch -> x'),
#             Linear(2 * 64, num_class),
#         ])
    