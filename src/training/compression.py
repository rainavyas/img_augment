from .trainer import Trainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, TensorDataset
from .pca import pca as do_pca
from datetime import datetime
import numpy as np
import pdb, torch
from torchvision.transforms import Resize
import torch.nn.functional as F


class CompressedDensitySampleTrainer(Trainer):
    '''
    Use a training set (e.g augmented training points) to learn a density distribution
    For a desired training set (e.g. without augmentation) calculate likelihood
    '''
    def __init__(self, ds_for_dist, train_ds, resize, pca, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0, components=100, size=32):
        super().__init__(device, model, optimizer, criterion, scheduler)

        # Compressed
        self.train_ds_comp = self.compress(train_ds, resize, size)
        self.ds_for_dist_comp = self.compress(ds_for_dist, resize, size)
        
        # Learn PCA projection matrix
        if pca:
            self.train_ds_pca_cls, self.train_ds_pca = do_pca(train_ds_comp, components = components)
            self.ds_for_dist_pca_cls, self.ds_for_dist_pca = do_pca(ds_for_dist_comp, components = components)

        # Learn distribution for p(x) and s(x), we will define t(x) = p(x)/s(x)
        self.dist_model = Estimator.train_kde(self.ds_for_dist_pca, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
        self.train_dist_model = Estimator.train_kde(self.train_ds_pca, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)

    def prep_weighted_dl(self, ds, gamma=1.0, bs=64):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by distribution model
        '''
        print("Getting weights", datetime.now())


        dist_weights = self.get_weights(self.dist_model, self.train_ds_cls.transform(self.train_ds_comp) if pca else self.train_ds_comp) # p(x)
        train_weights = self.get_weights(self.train_dist_model, self.ds_for_dist_cls.transform(self.train_ds_comp) if pca else self.train_ds_comp)# p(x)) # s(x)
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

    def apply_pca(ds, ds_cls):
        ys = []
        for i in range(len(ds)):
            _, y = ds[i]
            ys.append(y)

    def compress(self, ds, resize=False, size=32):
        if resize:
            xs = []
            ys = []
            for i in range(len(ds)):
                x, y = ds[i]
                xs.append(torch.Tensor(x))
                ys.append(y)
            pdb.set_trace()
            xs = torch.unsqueeze(torch.stack(xs), dim=1)
            out = F.interpolate(xs, size=size**2)
            ds = TensorDataset(torch.tensor(out), torch.LongTensor(ys))
        return ds
    
