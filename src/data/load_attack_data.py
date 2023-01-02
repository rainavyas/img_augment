import torch
from torch.utils.data import ConcatDataset

from .data_utils import train_selector, test_selector

def load_attacked(args, train=True, method='pgd', delta=0.1, only_adv=False):

    if train:
        
        # Load attacked
        fpath = f'{args.data_dir_path}/Attacked/{args.data_name}-{args.domain}_train_{method}_{args.model_name}_delta{delta}.pt'
        train_ds_a = torch.load(fpath)
        fpath = f'{args.data_dir_path}/Attacked/{args.data_name}-{args.domain}_val_{method}_{args.model_name}_delta{delta}.pt'
        val_ds_a = torch.load(fpath)

        if only_adv:
            return train_ds_a, val_ds_a

        # Load original
        args.prune = -1
        args.aug = False
        train_ds, val_ds = train_selector(args)

        return ConcatDataset(train_ds, train_ds_a), ConcatDataset(val_ds, val_ds_a)
    
    if not train:
        # load only attacked test data 
        fpath = f'{args.data_dir_path}/Attacked/{args.data_name}-{args.domain}_test_{method}_{args.model_name}_delta{delta}.pt'
        return torch.load(fpath)