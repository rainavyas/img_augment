import torch 
from..tools.tools import AverageMeter, accuracy_topk
from.trainer import Trainer
from torch.utils.data import TensorDataset
from.density_sampling import DensitySampleTrainer


class ModifiedLossTrainer(Trainer):
    '''
    All training functionality with loss modified by importance weights - p(x)/s(x)
    '''
    def __init__(self, device, model, optimizer, criterion, scheduler):
        super().__init__(device, model, optimizer, criterion, scheduler)

    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
        '''
        Run one train epoch
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to train mode
        model.train()

        for i, (x, y, w) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)*w

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, y)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tAccuracy {accs.val:.3f} ({accs.avg:.3f})')

    @staticmethod
    def calculate_importance_weights(ds_for_dist, train_ds, kernel='gaussian', bandwidth=1, kde_frac=1.0, gamma=1.0):
        '''
        Calculating p(x)/s(x)
        '''
        dist_model = Estimator.train_kde(ds_for_dist, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
        train_dist_model = Estimator.train_kde(train_ds, kernel=kernel, bandwidth=bandwidth, kde_frac=kde_frac)
        print("Getting weights", datetime.now())
        dist_weights = DensitySampleTrainer.get_weights(dist_model, train_ds) # p(x)
        train_weights = DensitySampleTrainer.get_weights(train_dist_model, train_ds) # s(x)
        corrected_weights = dist_weights / train_weights # p(x)/s(x) = w
        corrected_weights = corrected_weights**gamma # p(x)/s(x)**gamma
        print("Got weights", datetime.now())
        return corrected_weights
    
    
    @staticmethod
    def weight_in_dataset(ds, ws):  
        '''
        Add importance weights to existing datasets
        '''
        xs = []
        ys = []
        ws = torch.from_numpy(ws)
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(x)
            ys.append(y)        
        return TensorDataset(torch.stack(xs, dim=0), torch.stack(ys, dim=0), ws)
