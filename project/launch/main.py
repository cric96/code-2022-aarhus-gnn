import yaml
from input_manager import configure_arg_parser
import os

parser = configure_arg_parser()
args = parser.parse_args()

with open(args.config) as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)
    metadata = configuration['metadata']
    simulations = configuration['simulations']
    for simulation in simulations:
        metadata['min_delta'] = metadata['min_delta'] or args.min_delta or 0.00001
        metadata['patience'] = metadata['patience'] or args.patience or 2
        metadata['accelerator'] = args.accelerator or metadata['accelerator']
        metadata['data_size'] = args.data_size or metadata['data_size']
        metadata["data_path"] = args.data_path or metadata["data_path"]
        single_configuration = {
            'metadata': metadata,
            'simulation': simulation
        }
        file = open("dump.yml", "w")
        yaml.dump(single_configuration, file)
        file.close()
        os.system("python single_run.py dump.yml")
os.remove("dump.yml")
