from .trainer import Trainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime

class DensitySampleTrainer(Trainer):
    '''
    Use a training set (e.g augmented training points) to learn a density distribution
    For a desired training set (e.g. without augmentation) calculate likelihood
    '''
    def __init__(self, ds_for_dist, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0):
        super().__init__(device, model, optimizer, criterion, scheduler)

        # Learn distribution
        self.dist_model = Estimator.train_kde(ds_for_dist, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
    
    def prep_weighted_dl(self, ds, bs=64):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by distribution model
        '''
        print("Getting weights", datetime.now())
        weights = Estimator.test_kde(ds, self.dist_model)
        import pdb; pdb.set_trace()
        print("Got weights", datetime.now())
        sampler = WeightedRandomSampler(weights, len(ds), replacement=True)
        print("Done init sampler, creating dl", datetime.now())
        dl = DataLoader(ds, batch_size=bs, sampler=sampler)
        print("created dl", datetime.now())
        return dl