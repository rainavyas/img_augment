import torch
import torch.nn as nn
import sys
import os
import argparse
from statistics import mean, stdev

from src.tools.tools import get_default_device
from src.models.ensemble import Ensemble
from src.data.data_selector import data_sel

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path_base', type=str, required=True, help='e.g. experiments/trained_models/my_model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--num_seeds', type=int, default=1, help="Specify number of seeds for model to load")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--domain', type=int, default=0, help="Specify domain for test set")
    args = commandLineParser.parse_args()

    # Assume num seeds is one in this script
    model_paths = [f'{args.model_path_base}{i}.th' for i in range(args.num_seeds)]

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the test data
    ds = data_sel(args.data_name, args.data_dir_path, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False)

    # Load model
    ens_model = Ensemble(args.model_name, model_paths, device, num_classes=10)

    # Evaluate
    criterion = nn.CrossEntropyLoss().to(device)
    accs = ens_model.eval(dl, criterion, device)

    if len(args.model_paths) > 1:
        acc_mean = mean(accs)
        acc_std = stdev(accs)
        out_str = f'{len(args.model_paths)} models\nOverall {acc_mean:.3f}+-{acc_std:.3f}'
    print(out_str)