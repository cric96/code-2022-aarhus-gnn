import argparse
import pandas as pd
import torch
from data.loader import PhenomenaDataLoader
from model.linear import Linear
from model.sequential import SpatioTemporalConvolutionGru, SpatioTemporalConvolutionLstm, SpatialGNN, \
    TemporalGru, SpatialPlusTemporal
## For plotly visualization
import dash
from dash import dcc, Output, Input, State
from dash import html


parser = argparse.ArgumentParser(description='Evaluation of neural model')
parser.add_argument('path', metavar='path', type=str, help='The path of model to load')
parser.add_argument('-p', '--data-path', type=str, help='The path of the data to load for the training part')
parser.add_argument('-d', '--data-size', type=int, help='The path of the data to load for the training part')

args = parser.parse_args()

models = [
    SpatioTemporalConvolutionGru, SpatioTemporalConvolutionLstm, SpatialGNN, TemporalGru, Linear, SpatialPlusTemporal
]

for model in models:
    try:
        loaded = model.load_from_checkpoint(args.path)
        print("Load using " + str(model))
        break
    except Exception as exc:
        print(exc)
        loaded = None

if loaded is None:
    print("Problem in loading the file...")
    exit()

model = loaded
forecast_size = model.output_feature_size
data_size = args.data_size or 10
loader = PhenomenaDataLoader(args.data_path or "../data/raw/", data_size, forecast_size)
loader.clean_position()
torch_graph_data = loader.data

split_test = 0.8
split_validation = 0.8

split_index = int(data_size * split_test)
torch_graph_train, torch_graph_test = torch_graph_data[:split_index], torch_graph_data[split_index:]
split_index_val = int(len(torch_graph_train) * split_validation)
torch_graph_train, torch_graph_validation = torch_graph_train[:split_index_val], torch_graph_train[
                                                                                 split_index_val:]

data_to_check = torch_graph_test[0]


def evaluation_pass(snapshosts, model, forecast_size, memory_adjust=None):
    cost = 0
    center = memory_adjust
    trajectory = []
    h = None
    for (index, snapshot) in enumerate(snapshosts[:-forecast_size]):
        x = snapshot.x
        # if(center != None):
        #   x_center = snapshot.x[center]
        #   x = torch.zeros(snapshot.x.shape)
        #   x[center] = x_center
        (y_hat, h) = model(x, snapshot.edge_index, snapshot.edge_attr, h)
        #if center is not None and h is not None:
        #    zeros = torch.zeros(h.shape)
        #    zeros[center] = h[center]
        #    h = zeros
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
        trajectory.append(y_hat)
    return cost / (len(snapshosts) - forecast_size), trajectory


def trajectory_for(reference, sensor_index):
    return [int(graph[sensor_index][0].item() * 100) for graph in reference]


model.eval()
total_cost = 0
test = data_to_check[1:]
total_nodes = test[0].x.size()[0]
(cost, trajectory) = evaluation_pass(data_to_check, model, model.output_feature_size)
start, end = (0, 900)
node = 100

print("Total node = " + str(total_nodes))
ground_truth = [graph.x for graph in test[start:end]]
y = trajectory_for(ground_truth, node)
pd.options.plotting.backend = "plotly"
computed = trajectory_for(trajectory[start:end], node)
all_trajectories = [(trajectory_for(trajectory[start:end], node), trajectory_for(ground_truth, node)) for node in range(0, total_nodes)]
all_trajectories = [pd.DataFrame(dict(forecast=trajectory, y=y)) for (trajectory, y) in all_trajectories]
figs = [dataframe.plot.line() for dataframe in all_trajectories]
lines = pd.DataFrame(dict(forecast=computed, y=y))
all_graph = [dcc.Graph(figure=line) for line in figs]
#fig = lines.plot.line()

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(dcc.Input(id='input-on-submit', type='text')),
        html.Button('Submit', id='submit-val', n_clicks=0),
        dcc.Graph(id="figure", figure=figs[0])
])

@app.callback(
    Output('figure', 'figure'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value')
)
def update_output(n_clicks, value):
    return figs[int(value)]

if __name__ == '__main__':
    app.run_server(debug=False)
