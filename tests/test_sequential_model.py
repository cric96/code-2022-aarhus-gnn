import pytest
from torch.utils.data import DataLoader

from project.data.loader import PhenomenaDataLoader, GraphDatasetIterator
from project.model.sequential import SpatioTemporalConvolutionGru, SpatioTemporalConvolutionLstm, TemporalGru, SpatialGNN, SpatialPlusTemporal
from project.model.linear import Linear
import pytorch_lightning as pl
forecast_size = 3
loader = PhenomenaDataLoader("./snapshots/subset/", 1, forecast_size)
loader.clean_position()

testdata = [
    SpatioTemporalConvolutionGru(1, forecast_size, 5),
    SpatioTemporalConvolutionLstm(1, forecast_size, 5),
    Linear(1, forecast_size, 5),
    TemporalGru(1, forecast_size, 5),
    SpatialGNN(1, forecast_size, 5),
    SpatialPlusTemporal(1, forecast_size, 10, 5)
]
ids = ["spatio temporal gru", "spatio temporal lstm", "feed forward", "temporal gru", "spatial GNN", "spatia and temporal"]


@pytest.mark.parametrize("model", testdata, ids=ids)
def test_model_forward(model):
    snapshot = loader.data[0][0]
    (forward, memory) = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    assert forward.shape == (400, forecast_size)

@pytest.mark.parametrize("model", testdata, ids=ids)
def test_model_train(model):
    trainer = pl.Trainer(max_epochs=1)
    snapshots = loader.data[0]
    split_at = 40
    train, validation = snapshots[:split_at], snapshots[split_at:]
    trainer.fit(model, GraphDatasetIterator([train]), GraphDatasetIterator([validation]))
