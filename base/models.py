import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GATv2Conv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hid=8, dropout=0.6):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(num_features, hid)
        self.conv2 = GCNConv(hid, num_classes)
        self.sel = nn.Sequential(nn.Linear(num_classes, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hid=8, dropout=0.6):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout
        self.sage1 = SAGEConv(num_features, hid)
        self.sage2 = SAGEConv(hid, num_classes)
        self.sel = nn.Sequential(nn.Linear(num_classes, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x, edge_index):

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.sage1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hid=8, in_head=8, out_head=4, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout        
        self.conv1 = GATConv(num_features, hid, heads=in_head, dropout=dropout)
        self.conv2 = GATConv(hid*in_head, num_classes, concat=False,
                             heads=out_head, dropout=dropout)
        self.sel = nn.Sequential(nn.Linear(num_classes, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x, edge_index):

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, n_layers, num_features, num_classes, hid=8, in_head=8, out_head=4, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout        
        # self.conv1 = GATConv(num_features, hid, heads=in_head, dropout=dropout)
        # self.conv2 = GATConv(hid*in_head, num_classes, concat=False,
        #                      heads=out_head, dropout=dropout)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hid, heads=in_head, concat=True))
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hid * in_head, hid, heads=in_head, concat=True))
        self.convs.append(GATConv(hid * in_head, num_classes, heads=1, concat=False))
        self.sel = nn.Sequential(nn.Linear(num_classes, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)

        return F.log_softmax(x, dim=1)

class GATv2(torch.nn.Module):
    def __init__(self, num_features, num_classes, hid=8, in_head=8, out_head=4, dropout=0.6):
        super(GATv2, self).__init__()
        self.dropout = dropout        
        self.conv1 = GATv2Conv(num_features, hid, heads=in_head, dropout=dropout)
        self.conv2 = GATv2Conv(hid*in_head, num_classes, concat=False,
                             heads=out_head, dropout=dropout)
        self.sel = nn.Sequential(nn.Linear(num_classes, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x, edge_index):

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)