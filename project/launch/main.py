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
        metadata['min_delta'] = args.min_delta or metadata.get('min_delta') or 0.00001
        metadata['patience'] = args.patience or metadata.get('patience')  or 2
        metadata['accelerator'] = args.accelerator or metadata.get('accelerator')
        metadata['data_size'] = args.data_size or metadata.get('data_size')
        metadata["data_path"] = args.data_path or metadata.get("data_path")
        metadata['max_epochs'] = args.max_epochs or metadata.get('max_epochs')
        metadata['logger'] = args.logger_active or metadata.get("logger_active") or True
        single_configuration = {
            'metadata': metadata,
            'simulation': simulation
        }
        file = open("dump.yml", "w")
        yaml.dump(single_configuration, file)
        file.close()
        os.system("python project/launch/single_run.py dump.yml")
os.remove("dump.yml")
