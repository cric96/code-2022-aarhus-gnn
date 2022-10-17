from model.sequential import SpatioTemporalConvolutionGru, SpatioTemporalConvolutionLstm, TemporalGru, SpatialGNN
from model.linear import Linear
from pytorch_lightning.loggers import NeptuneLogger
from data.loader import PhenomenaDataLoader, GraphDatasetIterator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl


forecast_size = 3
data_size = 10
loader = PhenomenaDataLoader("../data/raw/", forecast_size)
loader.clean_position()
torch_graph_data_position = loader.data[:data_size]

neptune_logger = NeptuneLogger(
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZjMyMzc3Ni0yZDc4LTQzMWMtYTIzMi0wMDVlMDU5MWRiMDEifQ==",  # replace with your own
    project="cric96/gnn-forecast",
)

split_test = 0.8
split_validation = 0.8

torch_graph_data = remove_position_in_graph(torch_graph_data_position)
split_index = int(data_size * split_test)
torch_graph_train , torch_graph_test = torch_graph_data[:split_index], torch_graph_data[split_index:]
split_index_val = int(len(torch_graph_train) * split_validation)
torch_graph_train, torch_graph_validation = torch_graph_train[:split_index_val], torch_graph_train[split_index_val:]

early_stop_callback = EarlyStopping(monitor='val_loss',
                                    min_delta=0.00005,
                                    patience=5,
                                    verbose=False,
                                    mode='min',
                                    max_epoch=200)

spatio_gru = SpatioTemporalConvolutionGru(2, forecast_size, 32)
spatio_lstm = SpatioTemporalConvolutionLstm(2, forecast_size, 32)
spatio = SpatialGNN(1, forecast_size, 32)
temporal = TemporalGru(1, forecast_size, 32)
linear = Linear(1, forecast_size, 32)

networks = [spatio_gru, spatio_lstm, spatio, temporal, linear]

trainer = pl.Trainer(callbacks=[early_stop_callback], accelerator="gpu", devices=1, logger=neptune_logger)

for network in networks:
    trainer.fit(network, GraphDatasetIterator(torch_graph_train), GraphDatasetIterator(torch_graph_validation))
