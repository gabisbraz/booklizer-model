import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class GNN(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ("livro", "escrito_por", "autor"): SAGEConv((-1, -1), hidden_dim),
                ("autor", "escreveu", "livro"): SAGEConv((-1, -1), hidden_dim),
                ("livro", "tem_genero", "genero"): SAGEConv((-1, -1), hidden_dim),
                ("genero", "pertence_a", "livro"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="mean",
        )

        self.conv2 = HeteroConv(
            {
                ("livro", "escrito_por", "autor"): SAGEConv((-1, -1), out_dim),
                ("autor", "escreveu", "livro"): SAGEConv((-1, -1), out_dim),
                ("livro", "tem_genero", "genero"): SAGEConv((-1, -1), out_dim),
                ("genero", "pertence_a", "livro"): SAGEConv((-1, -1), out_dim),
            },
            aggr="mean",
        )

    def forward(self, data: HeteroData):
        # features iniciais = one-hot por tipo de nó
        x_dict = {
            ntype: torch.eye(num, dtype=torch.float)
            for ntype, num in data.num_nodes_dict.items()
        }

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: F.normalize(v, p=2, dim=1) for k, v in x_dict.items()}
        return x_dict


model_gnn = GNN(hidden_dim=24, out_dim=64)
