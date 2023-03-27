from .trainer import Trainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
import numpy as np
import pdb
from tqdm import tqdm

class DensitySampleTrainer(Trainer):
    '''
    Use a training set (e.g augmented training points) to learn a density distribution
    For a desired training set (e.g. without augmentation) calculate likelihood
    '''
    def __init__(self, ds_for_dist, train_ds, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0):
        super().__init__(device, model, optimizer, criterion, scheduler)
        
        # Learn distribution for p(x) and s(x), we will define t(x) = p(x)/s(x)
        self.dist_model = Estimator.train_kde(ds_for_dist, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
        self.train_dist_model = Estimator.train_kde(train_ds, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
    
    def prep_weighted_dl(self, ds, gamma=1.0, bs=64):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by distribution model
        '''
        print("Getting weights", datetime.now())
        dist_weights = self.get_weights(self.dist_model, ds) # p(x)
        train_weights = self.get_weights(self.train_dist_model, ds) # s(x)
        corrected_weights = dist_weights / train_weights # p(x)/s(x) = w
        corrected_weights = corrected_weights**gamma # p(x)/s(x)**gamma
        print("Got weights", datetime.now())
        sampler = WeightedRandomSampler(corrected_weights, len(ds), replacement=True)
        print("Done init sampler, creating dl", datetime.now())
        dl = DataLoader(ds, batch_size=bs, sampler=sampler)
        print("created dl", datetime.now())
        return dl
    
    @staticmethod 
    def get_weights(kde_dist, ds):
        #  return normalised likelihood for ds samples as per kde_dist
        log_weights = Estimator.test_kde(ds, kde_dist)
        scaled_log_weights = log_weights - np.max(log_weights)
        weights = np.exp(scaled_log_weights)
        weights = weights/np.sum(weights)
        return weights