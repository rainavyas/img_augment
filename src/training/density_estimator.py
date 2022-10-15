'''
Kernel Density Estimator - also known as the Parzen-Rosenblatt Window method
    It is a non-parametric method for estimating the probability density function of a given random variable

    Given i.i.d observations x_1, ..., x_n, then
        p(x) = \frac{1}{n}\sum_{j=1}^n K(\frac{x-x_j}{h}),

        where K() is a desired kernel function and h is our smoothing parameter/bandwidth.
'''

from sklearn.neighbors import KernelDensity
from torch.utils.data import TensorDataset, ConcatDataset, Subset
import torch
import numpy as np
import multiprocessing
from datetime import datetime


class Estimator():
    '''
    Uses Kernel Density Estimation for multi-dimensional Tensors
    ''' 

    @staticmethod
    def _process_ds_and_flatten(ds):
        '''
        Assume ds has x,y
        '''
        xs=[]
        for i in range(len(ds)):
            x, _ = ds[i]
            xs.append(torch.flatten(x))
        return torch.stack(xs, dim=0).cpu().detach().numpy()

    @classmethod
    def train_kde(cls, ds, kernel='gaussian', bandwidth=1, kde_frac=1.0):
        subset_idx = torch.randperm(len(ds))[:int(kde_frac*len(ds))]
        ds = Subset(ds,subset_idx)
        X = cls._process_ds_and_flatten(ds)
        model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        model.fit(X)
        return model
    
    @classmethod
    def test_kde(cls, test_ds, model):
        X = cls._process_ds_and_flatten(test_ds)
        # import pdb; pdb.set_trace()
        # log_prob = model.score_samples(X)
        # log_prob = cls.parrallel_score_samples(model, X)
        log_prob = cls.sequentially_score_samples(model, X)
        # scale probability densities (to get non-zero when we exponentiate)
        m = np.median(log_prob)
        log_prob -= m
        import pdb; pdb.set_trace()
        log_prob = np.clip(log_prob, None, 0.01*np.abs(m))
        return np.exp(log_prob)

    @staticmethod
    def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
        # currently failing due to pickle memory being exceeded by parallel threads
        # Solution: need to pickle save the output of each thread and then load all of them
        print(f"Using {thread_count} threads")
        samples = samples[:50] # temp
        with multiprocessing.Pool(thread_count) as p:
            return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

    @staticmethod
    def sequentially_score_samples(kde, samples, num=10):
        seq_samples = np.array_split(samples, num)
        scores = []
        for i,s in enumerate(seq_samples):
            print(f'On {i}/{num} {datetime.now()}')
            scores.append(kde.score_samples(s))
        return np.concatenate(scores)