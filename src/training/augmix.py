'''
augmix training approach:

- each image orig, augmented to create augmix1 and augmix2
- training: loss = CE(orig) + JSD_loss(orig, augmix1, augmix2)


Option to integrate with our df method
'''

from .single_density_sampling import SingleDensitySampleTrainer
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from..tools.tools import AverageMeter, accuracy_topk, print_log, get_ds_range

from datetime import datetime
import torch
# from torchvision.transforms import AugMix
from src.data.transforms import AugMix
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class AugMixTrainer(SingleDensitySampleTrainer):
    '''
    Use a training set to learn a density distribution
    apply dist_transform to train set likelihoods to obtain weights for training
    '''
    def __init__(self, train_ds, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0, df=False):
        SingleDensitySampleTrainer.__init__(self, train_ds, device, model, optimizer, criterion, scheduler, kernel, bandwidth, kde_frac, df=df)
    
    @staticmethod
    def augmix_ds(ds):
        '''
        input: ds (x,y)
        return: aug_ds (x, augmix1, augmix2, y)
        
        order is maintained
        '''
        transform = AugMix()
        xs = []
        aug1s = []
        aug2s = []
        ys = []
        print('generating augmix ds')
        small, big = get_ds_range(ds)
        small = small.unsqueeze(dim=-1).unsqueeze(dim=-1)
        big = big.unsqueeze(dim=-1).unsqueeze(dim=-1)
        r = big-small
        for i in tqdm(range(len(ds))):
            x, y = ds[i]
            x_scaled  = (((x - small)/r) * 255).to(torch.uint8) # values between 0-255 for transformation
            aug1s.append(((transform(x_scaled)/255)*r) + small)
            aug2s.append(((transform(x_scaled)/255)*r) + small)
            ys.append(y)
            xs.append(x)
        xs = torch.stack(xs, dim=0)
        aug1s = torch.stack(aug1s, dim=0)
        aug2s = torch.stack(aug2s, dim=0)
        ys = torch.LongTensor(ys)
        return TensorDataset(xs, aug1s, aug2s, ys)
    
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

    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
        '''
        Run one train epoch for augmix
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to train mode
        model.train()

        for i, (x, aug1, aug2, y) in enumerate(train_loader):
            x = x.to(device)
            aug1 = aug1.to(device)
            aug2 = aug2.to(device)
            y = y.to(device)

            # Forward pass
            logits_clean = model(x)
            logits_aug1 = model(aug1)
            logits_aug2 = model(aug2)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)
        
            # standard cross-entropy loss on clean image
            ce_loss = F.cross_entropy(logits_clean, y)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            jrd_loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            loss = ce_loss + jrd_loss

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc = accuracy_topk(logits_clean.data, y)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

            if i % print_freq == 0:
                print_log(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tAccuracy {accs.val:.3f} ({accs.avg:.3f})')
    

class AugMix2Trainer(AugMixTrainer):
    '''
    Use a training set to learn a density distribution
    apply dist_transform to augmix train set likelihoods to obtain weights for training
    '''
    def __init__(self, augmix_ds, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0, df=False):
        AugMixTrainer.__init__(self, self.flatten_augmix_ds(augmix_ds), device, model, optimizer, criterion, scheduler, kernel, bandwidth, kde_frac, df=df)

    
    @staticmethod
    def flatten_augmix_ds(augmix_ds):
        '''
        Input: each sample in augmix_ds is (x, augmix1, augmix2, y)
        Output: each above sample becomes 3 samples: (x, y), (augmix1, y), (augmix2, y)
        '''

        xs = []
        ys = []
        for i in range(len(augmix_ds)):
            x, aug1, aug2, y = augmix_ds[i]
            xs += [x, aug1, aug2]
            ys += [y, y, y]
        xs = torch.stack(xs, dim=0)
        ys = torch.LongTensor(ys)
        return TensorDataset(xs, ys)

    def prep_weighted_dl(self, aug_ds, dist_transform='unity', gamma=1.0, bs=64, transform_args=None):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by dist_transform on likelihoods
        '''
        print("Getting weights", datetime.now())
        tw = self.get_weights(self.train_dist_model, self.flatten_augmix_ds(aug_ds)) # s(x)
        
        # unflatten weights and multiply: p(x)*p(aug1)*p(aug2)
        tw_splits = [split.squeeze() for split in np.vsplit(np.expand_dims(tw, axis=1), len(aug_ds))]
        train_weights = np.asarray([split[0]*split[1]*split[2] for split in tw_splits])
        train_weights = train_weights/np.sum(train_weights) # normalize

        transformed_weights = self.apply_transform(train_weights, dist_transform, transform_args=transform_args) # f(s(x)) = p(x), where p(x) is desired distribution
        corrected_weights = transformed_weights / train_weights # p(x)/s(x) = w
        corrected_weights = corrected_weights**gamma # p(x)/s(x)**gamma
        print("Got weights", datetime.now())
        sampler = WeightedRandomSampler(corrected_weights, len(aug_ds), replacement=True)
        print("Done init sampler, creating dl", datetime.now())
        dl = DataLoader(aug_ds, batch_size=bs, sampler=sampler)
        print("created dl", datetime.now())
        return dl
    
