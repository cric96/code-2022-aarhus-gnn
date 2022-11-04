from jedi.plugins import pytest
import torch
from data.loader import PhenomenaDataLoader

nodes = 400
simulations_snapshots = 100


def test_load_simple_data():
    file_size_in_folder = 4
    loader = PhenomenaDataLoader("./snapshots/subset/")
    assert len(loader.data) == file_size_in_folder


def test_loading_limit():
    loader = PhenomenaDataLoader("./snapshots/subset/", 1)
    assert len(loader.data) == 1


def test_forecast_window_size():
    window = 5
    loader = PhenomenaDataLoader("./snapshots/subset/", 1, window)
    snapshot = loader.data[0][0]  # first snapshot
    assert snapshot.y.shape == (nodes, window)


def test_clean_position():
    loader = PhenomenaDataLoader("./snapshots/subset/", 1)
    loader.clean_position()
    snapshot = loader.data[0][0]
    assert snapshot.x.shape == (nodes, 1)


def test_memory_window():
    window = 5
    loader = PhenomenaDataLoader("./snapshots/subset/", 1)
    dataset = loader.fixed_window_size(window)
    snapshot = dataset[0]
    assert snapshot.x.shape == (nodes, window)
    assert len(dataset) == len(loader.data[0]) - window
    assert torch.equal(dataset[0].y, loader.data[0][5].y)
