
import argparse
from pydoc import locate

import pytorch_lightning as pl
import yaml
from data.loader import PhenomenaDataLoader, GraphDatasetIterator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

parser = argparse.ArgumentParser(description='Evaluation of neural model')
parser.add_argument('config', metavar='config path', type=str, help='The path of the configuration file')
args = parser.parse_args()

with open(args.config) as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)
    metadata = configuration['metadata']
    simulation = configuration['simulation']

    print("Training of " + simulation['name'])
    forecast_size = simulation['args'][1]  ## forecast_size from args
    data_size = metadata['data_size']
    loader = PhenomenaDataLoader(metadata["data_path"], data_size, forecast_size)
    loader.clean_position()
    torch_graph_data = loader.data

    split_test = 0.8
    split_validation = 0.8

    split_index = int(data_size * split_test)
    torch_graph_train, torch_graph_test = torch_graph_data[:split_index], torch_graph_data[split_index:]
    split_index_val = int(len(torch_graph_train) * split_validation)
    torch_graph_train, torch_graph_validation = torch_graph_train[:split_index_val], torch_graph_train[
                                                                                     split_index_val:]

    full_classpath = simulation['module'] + "." + simulation['class_name']
    init = locate(full_classpath)
    network = init(*simulation['args'])

    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZjMyMzc3Ni0yZDc4LTQzMWMtYTIzMi0wMDVlMDU5MWRiMDEifQ==",
        project="PS-Lab/gnn-forecast",
        name=simulation['name'],
        description=simulation['description'],
        tags=simulation['tags']
    ) if metadata['logger_active'] else None

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=metadata['min_delta'],
        patience=metadata['patience'],
        verbose=True,
        mode='min'
    )
    print(logger)
    trainer = pl.Trainer(
        callbacks=[early_stop_callback],
        accelerator=metadata['accelerator'],
        devices=1,
        logger=logger,
        max_epochs=metadata['max_epochs']
    )
    train, validation = GraphDatasetIterator(torch_graph_train[:1]), GraphDatasetIterator(torch_graph_validation[:1])
    lr_finder = trainer.tuner.lr_find(network, train, validation, mode="linear")
    train, validation = GraphDatasetIterator(torch_graph_train), GraphDatasetIterator(torch_graph_validation)
    new_lr = lr_finder.suggestion()
    network.hparams.learning_rate = new_lr
    print("tuning ...")
    trainer.fit(network, train, validation)
    logger.finalize("success") if metadata['logger_active'] else None
