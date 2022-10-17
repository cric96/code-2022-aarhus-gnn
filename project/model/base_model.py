from abc import ABC
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from project.model.tensor_types import *

class BaseSpatioTemporal(ABC, pl.LightningModule):
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size):
        super().__init__()
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.hidden_feature_size = hidden_feature_size
        self.recurrent = self.init_spatio_temporal_layer()
        self.learning_rate = 0.01

    def init_spatio_temporal_layer(self) -> torch.nn.Module:
        pass

    def __simulation_pass__(self, batch) -> MetricCollection:
        pass

    def training_step(self, batch, batch_idx) -> MetricCollection:
        loss = self.__simulation_pass__(batch)
        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> MetricCollection:
        loss = self.__simulation_pass__(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class BaseSequentialSpatioTemporal(BaseSpatioTemporal):
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size):
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)
        self.linear = torch.nn.Linear(hidden_feature_size, output_feature_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, memory: Union[Tensor, None, tuple[Tensor]] = None) \
            -> (Tensor, Tensor):
        y, memory = self.__recurrent_pass__(x, edge_index, edge_weight, memory)
        y = F.relu(y)
        y = self.linear(y)
        return y, memory

    def __simulation_pass__(self, batch):
        cost = 0
        memory = None
        for snapshot in batch:
            x = snapshot.x
            y = snapshot.y
            (h, memory) = self(x, snapshot.edge_index, snapshot.edge_attr, memory)
            cost = cost + F.mse_loss(h, y)
        loss = cost / len(batch)
        return loss

    def __recurrent_pass__(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                           memory: Union[Tensor, None] = None) -> (Tensor, Tensor):
        h = self.recurrent(x, edge_index, edge_weight, memory)
        return h, h


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  ## todo move in another position
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }

class BaseWindowSpatioTemporal(BaseSpatioTemporal):
    def __init__(self, input_feature_size, output_feature_size, hidden_feature_size):
        super().__init__(input_feature_size, output_feature_size, hidden_feature_size)
        self.linear = torch.nn.Linear(hidden_feature_size, output_feature_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, memory: Union[Tensor, None, tuple[Tensor]] = None) \
            -> (Tensor, Tensor):
        y, memory = self.__recurrent_pass__(x, edge_index, edge_weight, memory)
        y = F.relu(y)
        y = self.linear(y)
        return y, memory

    def __simulation_pass__(self, batch):
        x = batch.x
        y = batch.y.view(-1, 1)
        h = self(x, batch.edge_index, batch.edge_attr)
        loss = F.mse_loss(h, y)
        return loss
