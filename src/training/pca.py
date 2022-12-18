from .trainer import Trainer
from .density_estimator import Estimator
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
import numpy as np
import pdb
import torch
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset

def pca(ds, components=10):
    xs = []
    ys = []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(torch.Tensor(x))
        ys.append(y)
    xs = torch.stack(xs)
    xs = xs.view(len(xs), xs[0].shape[-1]*xs[0].shape[-2]*xs[0].shape[-3])
    pca = PCA(n_components=components)
    reduced_pca = pca.fit_transform(xs)
    # pdb.set_trace()
    return TensorDataset(torch.tensor(reduced_pca), torch.LongTensor(ys))
    

