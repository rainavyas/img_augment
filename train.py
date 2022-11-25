import torch
import torch.nn as nn
import sys
import os, pdb
import argparse
from datetime import datetime
from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import model_sel
from src.data.data_selector import data_sel
from src.training.trainer import Trainer
from src.training.modified_loss_trainer import ModifiedLossTrainer
from src.training.density_sampling import DensitySampleTrainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=200, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.001, help="Specify learning rate")
    commandLineParser.add_argument('--momentum', type=float, default=0.9, help="Specify momentum")
    commandLineParser.add_argument('--weight_decay', type=float, default=1e-4, help="Specify momentum")
    commandLineParser.add_argument('--sch', type=int, default=[100, 150], nargs='+', help="Specify scheduler cycle")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--aug', action='store_true', help='use data augmentation')
    commandLineParser.add_argument('--aug_sample', action='store_true', help='use data augmentation to define a distribution and use this to sample original training samples')
    commandLineParser.add_argument('--kde_frac', type=float, default=1.0, help="Specify frac of data to keep for training kde estimator")
    commandLineParser.add_argument('--domain', type=str, default='none', help="Specify source domain for DA dataset")
    commandLineParser.add_argument('--prune', type=float, default=0.0, help="Specify pruning fraction")
    commandLineParser.add_argument('--only_aug', action='store_true', help='use only augmented data for target dist, otherwise orig+aug data for target')
    commandLineParser.add_argument('--B', type=float, default=1.0, help="KDE bandwidth")
    commandLineParser.add_argument('--aug_num', type=int, default=3, help="Number of times to augment")
    commandLineParser.add_argument('--gamma', type=float, default=1.0, help="Importance weighting power")
    commandLineParser.add_argument('--loss_imp_train', action='store_true', help='scale the loss by importance weights during training')

    args = commandLineParser.parse_args()

    set_seeds(args.seed)
    out_file = f'{args.out_dir}/{args.model_name}_{args.data_name}_{args.domain}_aug{args.aug}_aug-sample{args.aug_sample}_gamma{args.gamma}_only-aug_{args.only_aug}_B{args.B}_prune{args.prune}_kdefrac{args.kde_frac}_aug-num{args.aug_num}_loss_imp_train{args.loss_imp_train}_seed{args.seed}.th'

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Initialise model
    model = model_sel(args.model_name)
    model.to(device)

    # Define learning objects
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Current time: ", datetime.now())
    # Load the training data and construct Trainer
    if args.aug_sample or args.loss_imp_train:
        # load augmented train data
        args.aug = True
        ds_for_dist, _ = data_sel(args, train=True, only_aug=args.only_aug)
        
        # load non-augmented train and val data
        args.aug = False
        train_ds, val_ds = data_sel(args, train=True)

        if args.aug_sample:
            trainer = DensitySampleTrainer(ds_for_dist, train_ds, device, model, optimizer, criterion, scheduler, kde_frac = args.kde_frac, bandwidth=args.B)
            train_dl = trainer.prep_weighted_dl(train_ds, gamma=args.gamma, bs=args.bs)
        else:
            # modified loss by importance weights
            trainer = ModifiedLossTrainer(device, model, optimizer, criterion, scheduler)
            weights = trainer.calculate_importance_weights(ds_for_dist, train_ds, bandwidth=args.bandwidth, kde_frac=args.kde_frac, gamma=args.gamma)
            train_ds = trainer.weight_in_dataset(train_ds, weights)
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    else:
        train_ds, val_ds = data_sel(args, train=True)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)
        trainer = Trainer(device, model, optimizer, criterion, scheduler)


    # Train
    trainer.train_process(train_dl, val_dl, out_file, max_epochs=args.epochs)