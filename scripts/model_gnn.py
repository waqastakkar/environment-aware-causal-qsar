from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function
from torch_geometric.nn import GINEConv, GPSConv, GraphNorm, TransformerConv, global_mean_pool


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_grl * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_grl: float = 1.0) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_grl)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        encoder: str = "gine",
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.encoder = encoder
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if encoder == "gine":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                conv = GINEConv(mlp, edge_dim=edge_dim, train_eps=True)
            elif encoder == "graph_transformer":
                local = TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    beta=True,
                )
                conv = GPSConv(hidden_dim, local, heads=4)
            else:
                raise ValueError(f"Unsupported encoder: {encoder}")
            self.layers.append(conv)
            self.norms.append(GraphNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h, batch)
            h = torch.relu(h)
            h = self.dropout(h)
        return global_mean_pool(h, batch)


class CausalQSARModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        z_dim: int,
        z_inv_dim: int,
        z_spu_dim: int,
        n_envs: int,
        task: str,
        encoder: str = "gine",
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.backbone = GNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=z_dim,
            encoder=encoder,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.f_inv = nn.Sequential(nn.Linear(z_dim, z_inv_dim), nn.ReLU(), nn.Linear(z_inv_dim, z_inv_dim))
        self.f_spu = nn.Sequential(nn.Linear(z_dim, z_spu_dim), nn.ReLU(), nn.Linear(z_spu_dim, z_spu_dim))
        self.predictor = nn.Sequential(nn.Linear(z_inv_dim, z_inv_dim), nn.ReLU(), nn.Linear(z_inv_dim, 1))
        self.grl = GradientReversal(1.0)
        self.adversary = nn.Sequential(nn.Linear(z_inv_dim, z_inv_dim), nn.ReLU(), nn.Linear(z_inv_dim, n_envs))

    def forward(self, data, lambda_grl: float = 1.0):
        self.grl.lambda_grl = lambda_grl
        h = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
        z_inv = self.f_inv(h)
        z_spu = self.f_spu(h)
        yhat = self.predictor(z_inv).squeeze(-1)
        envhat = self.adversary(self.grl(z_inv))
        return {
            "h": h,
            "z_inv": z_inv,
            "z_spu": z_spu,
            "yhat": yhat,
            "envhat": envhat,
        }
