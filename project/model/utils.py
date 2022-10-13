from typing import Tuple, Any

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
