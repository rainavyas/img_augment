import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset

class Attacker():
    @staticmethod
    def gradient(model, x, y, device, use_pred_label=True):
        '''
        Return gradient of loss wrt to input x
        '''
        x = x.to(device)
        model.eval()

        x.requires_grad = True
        y_pred = model(torch.unsqueeze(x, 0)).squeeze(0)
        
        if use_pred_label:
            # use predicted label to calculate loss wrt to
            y = torch.argmax(y_pred).item()
            
        loss = -1*torch.log(y_pred[y])
        loss.backward()
        direction = x.grad
        x.requires_grad = False
        return y_pred.squeeze(0).cpu().detach(), direction.cpu().detach()

    @classmethod
    def fgsm(cls, model, x, delta, device):
        '''FGSM attack'''
        _, direction = cls.gradient(model, x, None, device)
        sign = torch.sign(direction)
        x_attack = x+(delta*sign)
        return x_attack.cpu().detach()

    @classmethod
    def pgd(cls, model, x, delta, device, num_iter=5):
        '''PGD attack'''
        y_pred, gradient = cls.gradient(model, x, None, device)
        x = x.to(device)
        x_attack = x.clone()
        for _ in range(num_iter):
            gradient = torch.sign(gradient) * delta # force gradient to be a fixed size in l-inf norm
            gradient = gradient.to(device)
            x_attack += gradient
            x_attack =  torch.max(torch.min(x_attack, x+delta), x-delta) # project back into l-inf ball
            _, gradient = cls.gradient(model, x_attack, torch.argmax(y_pred).item(), device, use_pred_label=False)
        return x_attack.cpu().detach()
    
    @classmethod
    def attack(cls, x, model, device, method='pgd', delta=0.2):
        '''Adversarial attack'''
        if method == 'fgsm':
            return cls.fgsm(model, x, delta, device)
        elif method == 'pgd':
            return cls.pgd(model, x, delta, device)
    
    @classmethod
    def attack_ds(cls, ds, model, device, method='pgd', delta=0.2):
        '''Attack entire ds'''
        xs = []
        ys = []
        for i in tqdm(range(len(ds))):
            (x, y) = ds[i]
            xs.append(cls.attack(x, model, device, method=method, delta=delta))
            ys.append(y)
        xs = torch.stack(xs, dim=0)
        ys = torch.LongTensor(ys)
        return TensorDataset(xs, ys)
    
