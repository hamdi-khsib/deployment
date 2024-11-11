# gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class DeepReasoningGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout_prob=0.5):
        super(DeepReasoningGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.fc_input = torch.nn.Linear(in_channels, hidden_channels)
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.fc_definition = torch.nn.Linear(hidden_channels, out_channels)
        self.fc_synonym = torch.nn.Linear(hidden_channels, out_channels)
        self.fc_relation = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.fc_input(x))
        x = self.dropout(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        out_def = self.fc_definition(x)
        out_syn = self.fc_synonym(x)
        out_rel = self.fc_relation(x)
        return out_def, out_syn, out_rel
