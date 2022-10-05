import os
import json
import torch
from torch import Tensor
from torch import LongTensor
from torch_geometric.data import Data
import numpy as np


class PhenomenaDataLoader:
    def __init__(self, load_from: str, limit: int = None, forecast_size: int = 1):
        raw = self.__load_json(load_from, limit)
        self.data = [self.__covert_to_geometric(snapshot, forecast_size) for snapshot in raw]

    def clean_position(self):
        torch_graph_data = []
        for snapshots in self.data:
            flatten_snapshot = []
            for snapshot in snapshots:
                flatten_snapshot.append(
                    Data(snapshot.x[:, 0:1], edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr,
                         y=snapshot.y)
                )
            torch_graph_data.append(flatten_snapshot)
        self.data = torch_graph_data

    # NB! force the dataset to be cleaned
    def fixed_window_size(self, window_size):
        self.clean_position()

        def transform_fixed_memory(dataset):
            new_dataset = []
            for index, snapshot in enumerate(dataset[window_size:]):
                index = index + window_size
                X = list(map(lambda snapshot: snapshot.x, dataset[index - window_size:index]))
                X = torch.stack(X)
                (snapshots, nodes, _) = X.shape
                new_dataset.append(Data(x=X.reshape(nodes, snapshots), y=snapshot.y, edge_index=snapshot.edge_index,
                                        edge_attr=snapshot.edge_attr))
            return new_dataset

        return [flat for snapshots in self.data for flat in transform_fixed_memory(snapshots)]

    def __covert_to_geometric(self, raw, forecast_size):
        def convert_single_time_stamp(json):
            x = Tensor(json[0]).reshape((len(json[0]), 3))
            edge_info = np.array(json[1])
            edge_index = LongTensor(edge_info[:, :2]).t()
            edge_data = Tensor(edge_info[:, 2:])
            return Data(x=x, edge_index=edge_index, edge_attr=edge_data)

        def inject_forecast(current, next):
            forecast = [snapshot.x[:, 0] for snapshot in next]
            forecast = torch.stack(forecast)
            (node, feature) = forecast.shape
            forecast = forecast.reshape(feature, node)
            return Data(x=current.x, edge_index=current.edge_index, edge_attr=current.edge_attr, y=forecast)

        all_snapshots = [convert_single_time_stamp(snapshot) for snapshot in raw]
        return [inject_forecast(snapshot, all_snapshots[index + 1:index + 1 + forecast_size]) for index, snapshot in
                enumerate(all_snapshots[:-forecast_size])]

    def __load_json(self, load_from, limit):
        files = os.listdir(load_from)
        files = files if limit is None else files[:limit]
        graph_data = []
        for file_name in files:
            file = open(load_from + file_name)
            graph_data.append(json.load(file))
            file.close()
        return graph_data
