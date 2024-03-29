import torch
import random
import logging

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
      curr_ind = pred[:,curr_k]
      num_eq = torch.eq(curr_ind, target).sum()
      acc = num_eq/len(output)
      res_total += acc
    return res_total*100

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def print_log(out_str):
    print(out_str)
    try:
        logging.info(out_str)
    except:
        pass

def get_ds_range(ds):
        '''
        return the biggest and smallest values per channel in a dataset
        '''
        xs = []
        for i in range(len(ds)):
            x, _ = ds[i]
            xs.append(x)
        xs = torch.stack(xs, dim=0) # [B x 3 x H x W]
        xs_flat = torch.flatten(torch.transpose(xs, 0,1), start_dim=1) # [3 x B*H*W]
        small = torch.min(xs_flat, dim=1)[0] # [3]
        large = torch.max(xs_flat, dim=1)[0] # [3]
        return small, large
