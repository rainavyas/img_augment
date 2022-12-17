'''
general analysis script
'''

import sys
import os, pdb
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.training.density_estimator import Estimator
from src.data.data_selector import data_sel

def get_weights(kde_dist, ds):
    #  return normalised likelihood for ds samples as per kde_dist
    log_weights = Estimator.test_kde(ds, kde_dist)
    scaled_log_weights = log_weights - np.max(log_weights)
    weights = np.exp(scaled_log_weights)
    weights = weights/np.sum(weights)
    return weights

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--weights_dist_learn', action='store_true', help='save weights distribution')
    commandLineParser.add_argument('--weights_dist_type', type=str, choices=['p', 's', 'both'], default='both', help='which weight distribution to save')
    commandLineParser.add_argument('--weights_dist_visualize', action='store_true', help='visualize saved weights distribution')
    commandLineParser.add_argument('--data_name', type=str, default='cifar10', help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, default='data', help='path to data directory, e.g. data')
    commandLineParser.add_argument('--domain', type=str, default='none', help="Specify source domain for DA dataset")
    commandLineParser.add_argument('--only_aug', action='store_true', help='use only augmented data for target dist, otherwise orig+aug data for target')
    commandLineParser.add_argument('--B', type=float, default=1.0, help="KDE bandwidth")
    commandLineParser.add_argument('--prune', type=float, default=0.0, help="if you want to prune")
    commandLineParser.add_argument('--save_dir', type=str, default='.', help="dir to save output file")
    commandLineParser.add_argument('--load_files', type=str, default=[], nargs='+', help="paths of files to load")
    commandLineParser.add_argument('--names', type=str, default=[], nargs='+', help="name for each file in legend")
    commandLineParser.add_argument('--plot_file', type=str, default='.', help="path to save plot")
    commandLineParser.add_argument('--xhigh', type=float, default=0.0, help="if you want to clip the plot for visualizaion")
    commandLineParser.add_argument('--xlow', type=float, default=0.0, help="if you want to clip the plot for visualizaion")
    commandLineParser.add_argument('--aug_num', type=int, default=3, help="Number of times to augment")
    commandLineParser.add_argument('--gamma', type=float, default=1.0, help="Importance weighting power")
    commandLineParser.add_argument('--kde_frac', type=float, default=1.0, help="Specify frac of data to keep for training kde estimator")
    
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyze.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    if args.weights_dist_learn:
        # get the weight (likelihood) of every training sample under target distribution

        # load augmented train data
        args.aug = True
        ds_for_dist, _ = data_sel(args, train=True, only_aug=args.only_aug)
    
        # load non-augmented train data
        args.aug = False
        train_ds, _ = data_sel(args, train=True)

        # get weights
        if args.weights_dist_type == 'both' or args.weights_dist_type == 'p':
            dist_model = Estimator.train_kde(ds_for_dist, kernel='gaussian', bandwidth=args.B, kde_frac=args.kde_frac)
            dist_weights = get_weights(dist_model, train_ds) # p(x)
        
        if args.weights_dist_type == 'both' or args.weights_dist_type == 's':
            train_dist_model = Estimator.train_kde(train_ds, kernel='gaussian', bandwidth=args.B, kde_frac=args.kde_frac)
            train_weights = get_weights(train_dist_model, train_ds) # s(x)
        
        if args.weights_dist_type == 'both':
            corrected_weights = dist_weights / train_weights # p(x)/s(x) = w
            corrected_weights = corrected_weights**args.gamma # p(x)/s(x)**gamma

        # save weights
        out_file = f'{args.save_dir}/weights_dist_{args.weights_dist_type}_domain{args.domain}_{args.data_name}_B{args.B}_gamma{args.gamma}'
        if args.weights_dist_type=='both':
            np.save(out_file, corrected_weights)
        elif args.weights_dist_type=='p':
            np.save(out_file, dist_weights)
        elif args.weights_dist_type=='s':
            np.save(out_file, train_weights)
        
    
    if args.weights_dist_visualize:
        # cumulative histogram plot of weights
        sns.set_style('darkgrid')

        for fpath, name in zip(args.load_files, args.names):
            weights = np.load(fpath)
            weights = np.sort(weights)
        
            plt.plot(weights, np.arange(weights.size)/weights.size, label=name)
        
        plt.legend(title='Gamma') # earlier Bandwidth
        plt.xlabel('Normalized Weight')
        plt.ylabel('Cumulative Density')
        if args.xhigh > 0.0:
            plt.xlim([args.xlow, args.xhigh])
        plt.savefig(args.plot_file, bbox_inches='tight')


