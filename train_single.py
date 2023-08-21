'''
Standard single domain generalisation approaches with an option for our method,
density flatten (df) method, applied on top.

Base Generalisation method integration with our df method

1) ERM:        vanilla training.
               df is applied to the training dataset to get weights for drawing training samples. weighted training.

2) augmix:     from the original paper... 
               Each orig sample is augmented to create: augmix1 and augmix2
               Further, a Jensen-Shannon loss is added in training to 
               enforce consistent predictions for orig, augmix1 and augmix2.

               loss = CE_loss(orig) + JSD_loss(orig, augmix1, augmix2)

               We apply df to orig to get importance weights, w, and thus we draw orig
               (and associated augmix1 and augmix2) as per these weights, to simulate the loss as:

               loss = w*(CE(orig) + JSD_loss(orig, augmix1, augmix2))

               here w = p(x_orig)/s(x_orig)

3) augmix2:    As above, but w = p(x_orig, x_augmix1, x_augmix2)/s(x_orig, x_augmix1, x_augmix2)
                where we will estimate that p(x_orig, x_augmix1, x_augmix2) = p(x_orig)*p(x_augmix1)*p(x_augmix2)
                (and similarly for s(...))

                Where we learn the distribution over the original dataset augmented with the augmix1 and augmix2 samples

4) augmix3:    
3) acvc:


'''

import torch
import torch.nn as nn
import sys
import os
import logging

import argparse
from datetime import datetime
from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import model_sel
from src.data.data_selector import data_sel
from src.training.trainer import Trainer
from src.training.single_density_sampling import SingleDensitySampleTrainer
from src.training.augmix import AugMixTrainer, AugMix2Trainer


def base_name_creator(args, dfargs):
    base_name = f'{args.base_method}_{args.model_name}_{args.data_name}_{args.domain}_seed{args.seed}'
    if dfargs.df:
        base_name = 'df_' + base_name + f'B{dfargs.B}_gamma{dfargs.gamma}_kdefrac{dfargs.kde_frac}_transform{dfargs.transform}'
        if dfargs.transform == 'tunity':
            base_name += f'_th{dfargs.th}'
    return base_name
        
if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. digits')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=200, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.001, help="Specify learning rate")
    commandLineParser.add_argument('--momentum', type=float, default=0.9, help="Specify momentum")
    commandLineParser.add_argument('--weight_decay', type=float, default=1e-4, help="Specify momentum")
    commandLineParser.add_argument('--sch', type=int, default=[100, 150], nargs='+', help="Specify scheduler cycle")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--domain', type=str, default='none', help="Specify source domain for DA dataset")
    commandLineParser.add_argument('--base_method', type=str, default='erm', choices=['erm', 'augmix', 'augmix2', 'augmix3'], required=False, help='Baseline single domain generalisation method')

    dfParser = argparse.ArgumentParser(description='density flattening (our) generalisation approach')
    dfParser.add_argument('--df', action='store_true', help='apply density flattening')
    dfParser.add_argument('--B', type=float, default=1.0, help="KDE bandwidth")
    dfParser.add_argument('--gamma', type=float, default=1.0, help=" exponent when aug sample ")
    dfParser.add_argument('--kde_frac', type=float, default=1.0, help="Specify frac of data to keep for training kde estimator")
    dfParser.add_argument('--transform', type=str, default='unity', choices=['unity', 'tunity', 'triangle'], required=False, help=' Transformation for s(x)')
    dfParser.add_argument('--th', type=float, default=0, help='Threshold for T-unity and Triangle')


    args, a = commandLineParser.parse_known_args()
    dfargs, s_a = dfParser.parse_known_args()
    assert set(a).isdisjoint(s_a), f"{set(a)&set(s_a)}"
    set_seeds(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_single.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    base_name = base_name_creator(args, dfargs)

    # Initialise logging
    if not os.path.isdir('LOGs'):
        os.mkdir('LOGs')
    fname = f'LOGs/{base_name}.log'
    logging.basicConfig(filename=fname, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('LOG created')

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
    train_ds, val_ds = data_sel(args, train=True)

    if args.base_method == 'erm':
        if dfargs.df:
            trainer = SingleDensitySampleTrainer(train_ds, device, model, optimizer, criterion, scheduler, kde_frac = dfargs.kde_frac, bandwidth=dfargs.B)
            train_dl = trainer.prep_weighted_dl(train_ds, dist_transform=dfargs.transform, gamma=dfargs.gamma, bs=args.bs, transform_args=dfargs)
        else:
            trainer = Trainer(device, model, optimizer, criterion, scheduler)
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    
    elif args.base_method == 'augmix':
        trainer = AugMixTrainer(train_ds, device, model, optimizer, criterion, scheduler, df=dfargs.df, bandwidth=dfargs.B, kde_frac=dfargs.kde_frac)
        aug_ds = trainer.augmix_ds(train_ds)
        if dfargs.df:
            train_dl = trainer.prep_weighted_dl(train_ds, aug_ds, dist_transform=dfargs.transform, gamma=dfargs.gamma, bs=args.bs, transform_args=dfargs)
        else:
            train_dl = torch.utils.data.DataLoader(aug_ds, batch_size=args.bs, shuffle=True)

    elif args.base_method == 'augmix2' or args.base_method == 'augmix3':
        aug_ds = AugMix2Trainer.augmix_ds(train_ds)
        trainer = AugMix2Trainer(aug_ds, device, model, optimizer, criterion, scheduler, df=dfargs.df, bandwidth=dfargs.B, kde_frac=dfargs.kde_frac)
        if dfargs.df:
            train_dl = trainer.prep_weighted_dl(aug_ds, dist_transform=dfargs.transform, gamma=dfargs.gamma, bs=args.bs, transform_args=dfargs, add = (args.base_method == 'augmix3') )
        else:
            train_dl = torch.utils.data.DataLoader(aug_ds, batch_size=args.bs, shuffle=True)
    
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)


            
    # Train
    out_file = f'{args.out_dir}/{base_name}.th'
    trainer.train_process(train_dl, val_dl, out_file, max_epochs=args.epochs)
