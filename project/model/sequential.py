import torch
from torch_geometric_temporal import GConvGRU

from project.model.base_model import BaseSequentialSpatioTemporal


class SpatioTemporalConvolutionGru(BaseSequentialSpatioTemporal):
    # K = 1 => No neighborhood, K>2 neighborhood
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size, K=1):
        self.K = K
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)

    def init_spatio_temporal_layer(self) -> torch.nn.Module:
        return GConvGRU(self.input_feature_size, self.hidden_feature_size, K=self.K)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)
