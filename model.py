import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn


class GraphCON(nn.Module):
    def __init__(self, GNNs, dt=1., alpha=1., gamma=1., dropout=None):
        super(GraphCON, self).__init__()
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.GNNs = GNNs  # list of the individual GNN layers
        self.dropout = dropout

    def forward(self, X0, Y0, edge_index):
        # set initial values of ODEs

        # solve ODEs using simple IMEX scheme
        for gnn in self.GNNs:
            Y0 = Y0 + self.dt * (torch.relu(gnn(X0, edge_index)) -
                                 self.alpha * Y0 - self.gamma * X0)
            X0 = X0 + self.dt * Y0

            if (self.dropout is not None):
                Y0 = F.dropout(Y0, self.dropout, training=self.training)
                X0 = F.dropout(X0, self.dropout, training=self.training)

        return X0, Y0


class deep_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dt=1., alpha=1., gamma=1., dropout=None):
        super(deep_GNN, self).__init__()
        self.enc = nn.Linear(nfeat, nhid)
        self.GNNs = nn.ModuleList()
        for _ in range(nlayers):
            self.GNNs.append(GCNConv(nhid, nhid))
        self.graphcon = GraphCON(self.GNNs, dt, alpha, gamma, dropout)
        self.dec = nn.Linear(nhid, nclass)

    def forward(self, x0, edge_index):
        # compute initial values of ODEs (encode input)
        x0 = self.enc(x0)
        # stack GNNs using GraphCON
        x0, _ = self.graphcon(x0, x0, edge_index)
        # decode X state of GraphCON at final time for output nodes
        x0 = self.dec(x0)
        return x0
