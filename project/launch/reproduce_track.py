import torch
import argparse

from project.model.base_model import BaseSequentialSpatioTemporal
from project.model.sequential import SpatioTemporalConvolutionGru, SpatioTemporalConvolutionLstm, SpatialGNN, TemporalGru
from project.model.linear import Linear
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml

from pydoc import locate

parser = argparse.ArgumentParser(description='Evaluation of neural model')
parser.add_argument('path', metavar='path', type=str, help='The path of model to load')
args = parser.parse_args()

models = [
    SpatioTemporalConvolutionGru, SpatioTemporalConvolutionLstm, SpatialGNN, TemporalGru, Linear
]


loaded = None
for model in models:
    try:
        loaded: BaseSequentialSpatioTemporal = model.load_from_checkpoint(args.path)
        print("Load using " + str(model))
        break
    except Exception as exc:
        print(exc)
        loaded = None

if(loaded == None):
    print("Problem in loading the file...")

with open("../config.yml") as file: ## todo add as argument
    simulations = yaml.load(file, Loader=yaml.FullLoader)
    for simulation in simulations:
        full_classpath = simulation['module'] + "." + simulation['class_name']
        init = locate(full_classpath)
        network = init(*simulation['args'])
        print(network)

