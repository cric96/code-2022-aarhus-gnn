from typing import Union

import torch
from torch import Tensor
from torch_geometric_temporal import GConvGRU, GConvLSTM
from model.utils import GnnTemporalWrap, GnnAndRnnWrap
from model.base_model import BaseSequentialSpatioTemporal
from torch_geometric.nn import GCN
from torch.nn import GRU

class SpatioTemporalConvolutionGru(BaseSequentialSpatioTemporal):
    # K = 1 => No neighborhood, K>2 neighborhood
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size, K=1):
        self.K = K
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)

    def __init_spatio_temporal_layer__(self) -> torch.nn.Module:
        return GConvGRU(self.input_feature_size, self.hidden_feature_size, K=self.K)


class SpatioTemporalConvolutionLstm(BaseSequentialSpatioTemporal):
    # K = 1 => No neighborhood, K>2 neighborhood
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size, K=1):
        self.K = K
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)

    def __init_spatio_temporal_layer__(self) -> torch.nn.Module:
        return GConvLSTM(self.input_feature_size, self.hidden_feature_size, K=self.K)

    def __recurrent_pass__(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                           memory: Union[Tensor, None] = None) -> (Tensor, Tensor):
        if memory is None:
            (h, c) = self.recurrent(x, edge_index, edge_weight)
            return h, (h, c)
        else:
            (h, c) = memory
            (h, c) = self.recurrent(x, edge_index, edge_weight, h, c)
            return h, (h, c)


class TemporalGru(BaseSequentialSpatioTemporal):
    # K = 1 => No neighborhood, K>2 neighborhood
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size, K=1):
        self.K = K
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)

    def __init_spatio_temporal_layer__(self) -> torch.nn.Module:
        return GnnTemporalWrap(torch.nn.GRU(self.input_feature_size, self.hidden_feature_size, batch_first=True))

    def __recurrent_pass__(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                           memory: Union[Tensor, None] = None) -> (Tensor, Tensor):
        return self.recurrent(x, edge_index, edge_weight, memory)


class SpatialGNN(BaseSequentialSpatioTemporal):
    # K = 1 => No neighborhood, K>2 neighborhood
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size, K=1):
        self.K = K
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)

    def __init_spatio_temporal_layer__(self) -> torch.nn.Module:
        return GCN(self.input_feature_size, self.hidden_feature_size, 1)

    def __recurrent_pass__(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                           memory: Union[Tensor, None] = None) -> (Tensor, Tensor):
        h = self.recurrent(x, edge_index, edge_weight)
        return (h, h)

class SpatialPlusTemporal(BaseSequentialSpatioTemporal):
    # K = 1 => No neighborhood, K>2 neighborhood
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size, temporal_feature_size):
        self.temporal_feature_size = temporal_feature_size
        super(SpatialPlusTemporal, self).__init__(input_feature_size, output_feature_size, hidden_feature_size)

    def __init_spatio_temporal_layer__(self) -> torch.nn.Module:
        gnn = GCN(self.input_feature_size, self.temporal_feature_size, 1)
        rnn = GRU(self.temporal_feature_size, self.hidden_feature_size, batch_first=True)
        return GnnAndRnnWrap(gnn, rnn)

    def __recurrent_pass__(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                           memory: Union[Tensor, None] = None) -> (Tensor, Tensor):
        return self.recurrent(x, edge_index, edge_weight)
