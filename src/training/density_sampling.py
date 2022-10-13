from .trainer import Trainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler

class DensitySampleTrainer(Trainer):
    '''
    Use a training set (e.g augmented training points) to learn a density distribution
    For a desired training set (e.g. without augmentation) calculate likelihood
    '''
    def __init__(self, ds_for_dist, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1):
        super().__init__(device, model, optimizer, criterion, scheduler)

        # Learn distribution
        self.dist_model = Estimator.train_kde(ds_for_dist, kernel=kernel, bandwidth=bandwidth)
    
    def prep_weighted_dl(self, ds, bs=64):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by distribution model
        '''
        weights = Estimator.test_kde(ds, self.dist_model)
        sampler = WeightedRandomSampler(weights, len(ds), replacement=True)
        dl = DataLoader(ds, batch_size=bs, sampler=sampler)
        return dl