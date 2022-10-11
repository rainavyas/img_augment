import torch, os, pdb
import numpy as np 
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data import random_split

def data_sel(args, train=True):
    if train ==True: return train_selector(args)
    else: return test_selector(args)


def train_selector(args, val=0.2):
    ds = tv.datasets.CIFAR10(root=args.data_dir_path, train=True, transform=transforms.Compose([
                    transforms.RandomCrop(size=32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ]), download=True)
    
    aug=args.aug
    if aug == True:
        for i in range(3):
            aug_ds = tv.datasets.CIFAR10(root=args.data_dir_path, train=True, transform=transforms.Compose([
                        transforms.AutoAugment(),
                        transforms.RandomCrop(size=32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010], 
                            ),
                        ]), download=True)
            if i == 0:
                full_ds = aug_ds
            else: 
                full_ds = ConcatDataset((full_ds, aug_ds))
        ds = ConcatDataset((full_ds, ds))
    num_val = int(val*len(ds))
    num_train = len(ds) - num_val
    train_ds, val_ds = random_split(ds, [num_train, num_val], generator=torch.Generator().manual_seed(42))
    return train_ds, val_ds

def test_selector(args):
    corr_path = os.path.join(args.data_dir_path, 'CIFAR-10-C')
    if args.domain == 0:
        test_ds = tv.datasets.CIFAR10(root=args.data_dir_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ]), download=True)
        return test_ds
    else:
        all_corr = []
        all_corr.append(np.load(os.path.join(corr_path, 'brightness.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'contrast.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'defocus_blur.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'elastic_transform.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'fog.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'frost.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'gaussian_blur.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'gaussian_noise.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'glass_blur.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'impulse_noise.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'jpeg_compression.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'motion_blur.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'pixelate.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'saturate.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'shot_noise.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'snow.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'spatter.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'speckle_noise.npy')))
        all_corr.append(np.load(os.path.join(corr_path, 'zoom_blur.npy')))
        labels = np.load(os.path.join(corr_path, 'labels.npy'))

        all_corr = torch.from_numpy(np.array(all_corr)).type(torch.FloatTensor)
        labels = torch.LongTensor(labels)

        corr = args.domain
        for i in range(19):
            curr_corr_dset = TensorDataset(all_corr[i][ (corr-1)*10000 : corr*10000], labels[(corr-1)*10000 : corr*10000] )
            if i == 0:
                all_corr_dset = curr_corr_dset
            else: 
                all_corr_dset = ConcatDataset((all_corr_dset, curr_corr_dset))
            
        # Apply Transforms to the datasets
        xs = []
        ys = []
        for i in range(len(all_corr_dset)):
            x, y = all_corr_dset[i]
            x = torch.permute(x, (2,0,1))
            x = x / 255
            T = transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
            x = T(x)
            xs.append(x)
            ys.append(y)
            

        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        all_corr_dset = TensorDataset(xs, ys)


        return all_corr_dset








        # brightness = np.load('brightness.npy')
        # contrast = np.load('contrast.npy')
        # defocus_blur = np.load('defocus_blur.npy')
        # elastic_transform = np.load('elastic_transform.npy')
        # fog = np.load('fog.npy')
        # frost = np.load('frost.npy')
        # gaussian_blur = np.load('gaussian_blur.npy')
        # gaussian_noise = np.load('gaussian_noise.npy')
        # glass_blur = np.load('glass_blur.npy')
        # impulse_noise = np.load('impulse_noise.npy')
        # jpeg_compression = np.load('jpeg_compression.npy')
        # labels = np.load('labels.npy')
        # motion_blur = np.load('motion_blur.npy')
        # pixelate = np.load('pixelate.npy')
        # saturate = np.load('saturate.npy')
        # shot_noise = np.load('shot_noise.npy')
        # snow = np.load('snow.npy')
        # spatter = np.load('spatter.npy')
        # speckle_noise = np.load('speckle_noise.npy')
        # zoom_blur = np.load('zoom_blur.npy')
        # labels = np.load('labels.npy')
            




