'''
Kernel Density Estimator - also known as the Parzen-Rosenblatt Window method
    It is a non-parametric method for estimating the probability density function of a given random variable

    Given i.i.d observations x_1, ..., x_n, then
        p(x) = \frac{1}{n}\sum_{j=1}^n K(\frac{x-x_j}{h}),

        where K() is a desired kernel function and h is our smoothing parameter/bandwidth.
'''

from sklearn.neighbors import KernelDensity
import torch
import numpy as np


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
    def train_kde(cls, train_ds, kernel='gaussian', bandwidth=1):
        X = cls._process_ds_and_flatten(train_ds)
        model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        model.fit(X)
        return model
    
    @classmethod
    def test_kde(cls, test_ds, model):
        X = cls._process_ds_and_flatten(test_ds)
        log_prob = model.score_samples(X)
        return np.exp(log_prob)
