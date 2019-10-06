import argparse
import yaml
import torch
from src import registry

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', help='catalyst config file', required=True)
parser.add_argument('--state', dest='state', help='catalyst checkpoint file', required=True)
parser.add_argument('--out', dest='out', help='file to use for storing traced model', required=True)
parser.add_argument('--shapes', dest='shape', help='shapes to use as input to a model', required=True, type=int, nargs='+')
parser.add_argument('--dtype', dest='dtype', help='tensor type', type=str, default='float')


def main():
    args = vars(parser.parse_args())
    config_file = args['config']
    model_file = args['state']
    out_file = args['out']
    shapes = args['shape']
    dtype = args['dtype']

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = registry.MODELS.get_from_params(**config['model_params'])
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if dtype == 'float':
        example_input = torch.randn(*shapes)
    elif dtype == 'int':
        example_input = torch.randint(0, 2, shapes)
    else:
        raise ValueError(f'\'dtype\'({dtype}) should be one of [int, float]')
    
    trace = torch.jit.trace(model, example_input)
    torch.jit.save(trace, out_file)
    print(f'Traced model {model_file} to {out_file}', flush=True)


if __name__ == '__main__':
    main()
