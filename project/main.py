from project.model.sequential import SpatioTemporalConvolutionGru, SpatioTemporalConvolutionLstm, TemporalGru, SpatialGNN
from project.model.linear import Linear
from pytorch_lightning.loggers import NeptuneLogger
from project.data.loader import PhenomenaDataLoader, GraphDatasetIterator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl


forecast_size = 2
window_size = 1
data_size = 10
loader = PhenomenaDataLoader("../data/raw/", data_size, forecast_size)
loader.clean_position()
torch_graph_data = loader.data

split_test = 0.8
split_validation = 0.8

split_index = int(data_size * split_test)
torch_graph_train , torch_graph_test = torch_graph_data[:split_index], torch_graph_data[split_index:]
split_index_val = int(len(torch_graph_train) * split_validation)
torch_graph_train, torch_graph_validation = torch_graph_train[:split_index_val], torch_graph_train[split_index_val:]

spatio_gru = SpatioTemporalConvolutionGru(window_size, forecast_size, 32)
spatio_lstm = SpatioTemporalConvolutionLstm(window_size, forecast_size, 32)
spatio = SpatialGNN(window_size, forecast_size, 32)
temporal = TemporalGru(window_size, forecast_size, 32)
linear = Linear(window_size, forecast_size, 32)

networks = [spatio_gru, spatio_lstm, spatio, temporal, linear]


for network in networks:
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.0001,
                                        patience=5,
                                        verbose=False,
                                        mode='min')
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZjMyMzc3Ni0yZDc4LTQzMWMtYTIzMi0wMDVlMDU5MWRiMDEifQ==",
        project="PS-Lab/gnn-forecast",
        name=str(type(network))
    )
    trainer = pl.Trainer(callbacks=[early_stop_callback], accelerator="gpu", devices=1, logger=neptune_logger,
                         max_epochs=200)
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(network, GraphDatasetIterator(torch_graph_train[:1]), GraphDatasetIterator(torch_graph_validation[:1]), mode="linear")
    new_lr = lr_finder.suggestion()
    # update hparams of the model
    network.hparams.learning_rate = new_lr
    print("tuning ...")
    trainer.fit(network, GraphDatasetIterator(torch_graph_train), GraphDatasetIterator(torch_graph_validation))
    neptune_logger.finalize("success")
