'''
augmix training approach:

- each image orig, augmented to create augmix1 and augmix2
- training: loss = CE(orig) + JSD_loss(orig, augmix1, augmix2)


Option to integrate with our df method
'''

from .single_denisty_sampling import SingleDensitySampleTrainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
import numpy as np
import pdb
from tqdm import tqdm

class AugMixTrainer(SingleDensitySampleTrainer):
    '''
    Use a training set to learn a density distribution
    apply dist_transform to train set likelihoods to obtain weights for training
    '''
    def __init__(self, train_ds, device, model, optimizer, criterion, scheduler, df=False, kernel='gaussian', bandwidth=1, kde_frac=1.0):
        super().__init__(train_ds, device, model, optimizer, criterion, scheduler, kernel, bandwidth, kde_frac, df=df)
    
    def augmix_ds(ds):
        '''
        input: ds (x,y)
        return: aug_ds (x, augmix1, augmix2, y)
        
        order is maintained
        '''
    
    def prep_weighted_dl(self, orig_ds, aug_ds, dist_transform='unity', gamma=1.0, bs=64, transform_args=None):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by dist_transform on likelihoods
        '''
        print("Getting weights", datetime.now())
        train_weights = self.get_weights(self.train_dist_model, orig_ds) # s(x)
        transformed_weights = self.apply_transform(train_weights, dist_transform, transform_args=transform_args) # f(s(x)) = p(x), where p(x) is desired distribution
        corrected_weights = transformed_weights / train_weights # p(x)/s(x) = w
        corrected_weights = corrected_weights**gamma # p(x)/s(x)**gamma
        print("Got weights", datetime.now())
        sampler = WeightedRandomSampler(corrected_weights, len(orig_ds), replacement=True)
        print("Done init sampler, creating dl", datetime.now())
        dl = DataLoader(aug_ds, batch_size=bs, sampler=sampler)
        print("created dl", datetime.now())
        return dl

        # need to rewrite train and test functions here
    
