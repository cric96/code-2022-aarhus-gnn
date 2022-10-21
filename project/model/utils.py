import torch

class GnnWrap(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.layer = module

    def forward(self, X: torch.FloatTensor, edge_index=None, edge_weight=None, memory=None) -> torch.Tensor:
        return self.layer(X)

class GnnTemporalWrap(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.layer = module

    def forward(self, X: torch.FloatTensor, edge_index=None, edge_weight=None, memory=None) -> tuple[torch.Tensor, torch.Tensor]:
        (h, memory) = self.layer(X.reshape(X.shape[0], 1, X.shape[1]), memory)
        return h.reshape(h.shape[0], h.shape[2]), memory


class GnnAndRnnWrap(torch.nn.Module):
    def __init__(self, gnn: torch.nn.Module, rnn: torch.nn.Module):
        super().__init__()
        self.spatio = gnn
        self.temporal = rnn

    def forward(self, X: torch.FloatTensor, edge_index=None, edge_weight=None, memory=None) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.spatio(X, edge_index, edge_weight)
        (s, memory_new) = self.temporal(s.view(s.shape[0], 1, s.shape[1]), memory)
        return s.view(s.shape[0], s.shape[2]), memory_new
