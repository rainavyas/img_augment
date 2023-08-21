from .trainer import Trainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
import numpy as np
import pdb
from tqdm import tqdm

class SingleDensitySampleTrainer(Trainer):
    '''
    Use a training set to learn a density distribution
    apply dist_transform to train set likelihoods to obtain weights for training
    '''
    def __init__(self, train_ds, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0, df=True):
        super().__init__(device, model, optimizer, criterion, scheduler)
        
        if df:
            # Learn distribution for s(x)
            self.train_dist_model = Estimator.train_kde(train_ds, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
    
    def prep_weighted_dl(self, ds, dist_transform='unity', gamma=1.0, bs=64, transform_args=None):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by dist_transform on likelihoods
        '''
        print("Getting weights", datetime.now())
        train_weights = self.get_weights(self.train_dist_model, ds) # s(x)
        transformed_weights = self.apply_transform(train_weights, dist_transform, transform_args=transform_args) # f(s(x)) = p(x), where p(x) is desired distribution
        corrected_weights = transformed_weights / train_weights # p(x)/s(x) = w
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
    
    @staticmethod
    def apply_transform(weights, dist_transform, transform_args=None):
        '''
        Transform the weights as per specified transformation to obtain weights for desired distribution
        '''
        if dist_transform == 'unity':
            # equiprobable distribution
            return np.ones((len(weights)))
        if dist_transform == 'tunity':
            # thresholded unity
            p = (weights>(transform_args.th/len(weights))).astype(int)
            return p

        if dist_transform == "triangle":
            # traingle unity
            th = transform_args.th/len(weights)
            p = np.clip(weights/th, a_min=0, a_max=1)
            return p
