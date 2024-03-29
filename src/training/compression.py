from .trainer import Trainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, TensorDataset
from .pca import pca as do_pca
from datetime import datetime
import numpy as np
import pdb, torch
from torchvision.models import resnet18
from torchvision.transforms import Resize
import torch.nn.functional as F
from tqdm import tqdm
from functools import lru_cache

class CompressedDensitySampleTrainer(Trainer):
    ''' 
    Use a training set (e.g augmented training points) to learn a density distribution
    For a desired training set (e.g. without augmentation) calculate likelihood
    '''
    def __init__(self, ds_for_dist, train_ds, resize, pca, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0, components=100, size=32, latent=False):
        super().__init__(device, model, optimizer, criterion, scheduler)

        self.resize = resize
        self.size = size
        self.pca = pca
        self.latent = latent
        self.ds_for_dist_pca_cls = None
        self.train_ds_pca_cls = None
        self.transform = pca or latent
        self.device = device

        # Compressed
        ds_for_dist_comp = self.compress(ds_for_dist, resize, size)
        train_ds_comp = self.compress(train_ds, resize, size)
        
        if self.latent:
            ds_for_dist_comp = self.latent_map(ds_for_dist_comp)
            train_ds_comp =self.latent_map(train_ds_comp)

        # Learn PCA projection matrix
        if pca:
            self.ds_for_dist_pca_cls, ds_for_dist_comp = do_pca(ds_for_dist_comp, components = components)
            self.train_ds_pca_cls, train_ds_comp = do_pca(train_ds_comp, components = components)

        # Learn distribution for p(x) and s(x), we will define t(x) = p(x)/s(x)
        self.dist_model = Estimator.train_kde(ds_for_dist_comp, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
        self.train_dist_model = Estimator.train_kde(train_ds_comp, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)

    def prep_weighted_dl(self, ds, gamma=1.0, bs=64):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by distribution model
        '''
        print("Getting weights", datetime.now()) 
        #  Resizing the original train_ds
        ds_comp = self.compress(ds, self.resize, self.size)

        dist_weights = self.get_weights(self.dist_model, self.apply_transform(ds_comp, self.ds_for_dist_pca_cls) if self.transform else ds_comp) # p(x)
        train_weights = self.get_weights(self.train_dist_model, self.apply_transform(ds_comp, self.train_ds_pca_cls) if self.transform else ds_comp) # s(x)
#         import pdb; pdb.set_trace()
        weights = dist_weights / train_weights # p(x)/s(x) = w
        corrected_weights = weights**gamma # p(x)/s(x)**gamma

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

    def apply_pca(self, ds, ds_cls):
        ys = []
        xs = []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(x)
            ys.append(y)
        xs = torch.stack(xs)
        modified_xs = xs.view(len(xs), xs[0].shape[-1]*xs[0].shape[-2]*xs[0].shape[-3])
        xs_pca = ds_cls.transform(modified_xs)
        return TensorDataset(torch.tensor(xs_pca), torch.LongTensor(ys))

    def compress(self, ds, resize=False, size=32):
        if resize:
            xs = []
            ys = []
            for i in range(len(ds)):
                x, y = ds[i]
                xs.append(torch.Tensor(x))
                ys.append(y)
            xs = torch.unsqueeze(torch.stack(xs), dim=1)
            out = F.interpolate(xs, size=size**2)
            ds = TensorDataset(torch.tensor(out), torch.LongTensor(ys))
        return ds
    
    def apply_transform(self, ds, ds_pca_cls=None):
        if self.latent:
            ds = self.latent_map(ds)
        if self.pca:
            ds = self.apply_pca(ds, ds_pca_cls)
        return ds

    @lru_cache(maxsize=8) 
    def latent_map(self, ds):
        # map every sample in dataset to resnet18 latent space

        dl = DataLoader(ds, batch_size=32, shuffle=False)
        xs = []
        ys = []
        for (x, y) in tqdm(dl):
            with torch.no_grad():
                x = x.to(self.device)
                model = resnet18(pretrained=True)
                model = torch.nn.Sequential(*(list(model.children())[:-2]))
                model.to(self.device)
                out = model(x).detach().cpu()
            xs.append(out)
            ys.append(y)
        x = torch.concat(xs, dim=0)
        y = torch.concat(ys, dim=0)

        return TensorDataset(x,y)
