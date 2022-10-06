from abc import ABC
import torch
from torch.nn import functional as F
import pytorch_lightning as pl


class BaseSpatioTemporal(ABC, pl.LightningModule):
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size):
        super().__init__()
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.hidden_feature_size = hidden_feature_size
        self.recurrent = self.init_spatio_temporal_layer()

        def init_spatio_temporal_layer(self) -> torch.nn.Module:
            pass

    def __simulation_pass__(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss = self.__simulation_pass__(batch)
        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.__simulation_pass__(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss


class BaseSequentialSpatioTemporal(ABC, BaseSpatioTemporal):
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size):
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)
        self.linear = torch.nn.Linear(hidden_feature_size, output_feature_size)

    def forward(self, x, edge_index, edge_weight, memory=None):
        memory = self.recurrent(x, edge_index, edge_weight, memory)
        h = F.relu(memory)
        h = self.linear(h)
        return h, memory

    def __simulation_pass__(self, batch):
        cost = 0
        memory = None
        for snapshot in batch:
            x = snapshot.x
            y = snapshot.y.view(-1, 1)
            (h, memory) = self(x, snapshot.edge_index, snapshot.edge_attr, memory)
            cost = cost + F.mse_loss(h, y)
        loss = cost / len(batch)
        return loss
