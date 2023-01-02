'''
Adversarial attack ds and save
'''

import sys
import os
import argparse
import torch

from src.tools.tools import get_default_device
from src.attack.attacker import Attacker
from src.data.data_selector import data_sel
from src.models.model_selector import model_sel

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='trained model path')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. digits')
    commandLineParser.add_argument('--domain', type=str, required=True, help="Specify source domain for DA dataset, e.g. usps")
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--attack_method', type=str, default='pgd', help="Specify attack method")
    commandLineParser.add_argument('--part', type=str, default='train', choices=['train', 'val', 'test'], help="Specify data split to attack")
    commandLineParser.add_argument('--delta', type=float, default=0.1, help="Specify perturbation size")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()
    
    # Load the data
    args.aug = False
    args.prune = -1
    if args.part == 'train':
        ds, val_ds = data_sel(args, train=True)
        if args.part == 'val':
            ds = val_ds
    else:
        ds = data_sel(args, train=True)


    # Load the model
    model = model_sel(args.model_name, args.model_path)
    model.to(device)

    # Attack
    attacked_ds = Attacker.attack_ds(ds, model, device, method=args.method, delta=args.delta)

    # Save
    if not os.path.isdir(f'{args.data_dir_path}/Attacked'):
        os.mkdir(f'{args.data_dir_path}/Attacked')
    out_file = f'{args.data_dir_path}/Attacked/{args.data_name}-{args.domain}_{args.part}_{args.method}_{args.model_name}_delta{args.delta}.pt'
    torch.save(attacked_ds, out_file)
