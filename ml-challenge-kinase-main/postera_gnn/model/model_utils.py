from torch_geometric.nn import MessagePassing

from torch import nn
from torch.nn import Linear, Parameter, init
from torch_geometric.nn import GCNConv, GINEConv, GINConv, GATConv
from torch_geometric.utils import add_self_loops, degree
#
def init_weights_xavier(m, gain=1.414, 
                        graph_layer_types=[GCNConv, GINEConv, GINConv]
    ):
    """
    Initialize weight with current best scheme (xavier)
    e.g.
        net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        net.apply(init_weights)
    """
    def init_weights(m, gain=gain):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=1)
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Linear):
        init_weights(m, gain=1)
  
    elif any(map(lambda l: isinstance(m, l), graph_layer_types)):
        m.nn.apply(init_weights)
        
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j