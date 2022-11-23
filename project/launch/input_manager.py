import argparse


def configure_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluation of neural model')
    parser.add_argument('config', metavar='config path', type=str, help='The path of the configuration file')
    parser.add_argument('-p', '--data-path', type=str, help='The path of the data to load for the training part')
    parser.add_argument('-d', '--data-size', type=int, help='How many simulations should the system consider')
    parser.add_argument('-e', '--max-epochs', type=int, help='Max number of epochs performed by the traning loop')
    parser.add_argument('-a', '--accelerator', type=str, help="Accelerator used for the training loop",
                        choices=['cpu', 'cuda', 'tpu'])
    parser.add_argument('-pa', '--patience', type=int, help="Patience used for early stopping")
    parser.add_argument('-m', '--min-delta', type=float, help="Min delta used for early stopping")
    return parser


