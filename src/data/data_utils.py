import torch, os, pdb
import numpy as np 
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, ConcatDataset, Subset
# from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

aug_transform = transforms.Compose([
                        transforms.AutoAugment(),
                        # transforms.RandomCrop(size=32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010], 
                            ),
                        ])
train_transform = transforms.Compose([
                    # transforms.RandomCrop(size=32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ])
test_transform = transforms.Compose([
                    transforms.Resize(16), # TODO comment out if you don't want to resize svhn - change for CIFAR as argument
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ])
test_transform_C = transforms.Compose([
                    #transforms.Resize(16),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ])

grayscale_aug_transform = transforms.Compose([
                        transforms.AutoAugment(),
                        # transforms.RandomCrop(size=32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Grayscale(3),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010], 
                            ),
                        ])
grayscale_test_transform = transforms.Compose([
                        transforms.Resize(16), # TODO comment out if you don't want to resize mnist - make arg to select if you want to resize
                        transforms.Grayscale(3),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010], 
                            ),
                        ])
grayscale_train_transform = transforms.Compose([
                        # transforms.RandomCrop(size=32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Grayscale(3),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010], 
                            ),
                        ])

def train_selector(args, val=0.2, only_aug=False):
    if args.data_name =='cifar10':
        ds = tv.datasets.CIFAR10(root=args.data_dir_path, train=True, transform= train_transform, download = True)
        if args.aug == True:
            for i in range(args.aug_num):
                aug_ds = tv.datasets.CIFAR10(root=args.data_dir_path, train=True, transform = aug_transform, download=True)
                if i == 0:
                    full_ds = aug_ds
                else: 
                    full_ds = ConcatDataset((full_ds, aug_ds))
            if only_aug:
                ds = full_ds
            else:
                ds = ConcatDataset((full_ds, ds))


    if args.data_name =='digits':
        mnist_path = os.path.join(args.data_dir_path, 'MNIST')
        usps_path = os.path.join(args.data_dir_path, 'usps.bz2')
        svhn_path = os.path.join(args.data_dir_path, 'train_32x32.mat')
        if args.domain == 'mnist': ds = tv.datasets.mnist.MNIST(args.data_dir_path, train=True, download=True, transform = grayscale_train_transform)
        elif args.domain == 'svhn': ds = tv.datasets.svhn.SVHN(args.data_dir_path, split='train', download=True, transform = train_transform)
        elif args.domain == 'usps': ds = tv.datasets.usps.USPS(args.data_dir_path, train=True, download=True, transform = grayscale_train_transform)
        if args.aug == True:
            for i in range(args.aug_num):
                if args.domain == 'mnist': aug_ds = tv.datasets.mnist.MNIST(args.data_dir_path, train=True, download=True, transform = grayscale_aug_transform)
                elif args.domain == 'svhn': aug_ds = tv.datasets.svhn.SVHN(args.data_dir_path, split='train', download=True, transform = aug_transform)
                elif args.domain == 'usps': aug_ds = tv.datasets.usps.USPS(args.data_dir_path, train=True, download=True, transform = grayscale_aug_transform)
                if i == 0:
                    full_ds = aug_ds
                else: 
                    full_ds = ConcatDataset((full_ds, aug_ds))
            if only_aug:
                ds = full_ds
            else:
                ds = ConcatDataset((full_ds, ds))

    if args.prune > 0:
        subset_idx = torch.randperm(len(ds))[:int(args.prune*len(ds))]
        ds = Subset(ds,subset_idx)


    num_val = int(val*len(ds))
    train_indices, val_indices = train_test_split(range(len(ds)), test_size=num_val, random_state=42)
    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)

    # num_train = len(ds) - num_val
    # train_ds, val_ds = random_split(ds, [num_train, num_val], generator=torch.Generator().manual_seed(42))
    print("Train data size", len(train_ds))
    print("Validation data size", len(val_ds))
    return train_ds, val_ds

def test_selector(args):
    if args.data_name =='cifar10':
        args.domain = int(args.domain)
        corr_path = os.path.join(args.data_dir_path, 'CIFAR-10-C')
        if args.domain == 0:
            test_ds = tv.datasets.CIFAR10(root=args.data_dir_path, train=False, transform=test_transform_C, download=True)
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


    elif args.data_name == 'digits':
        if args.domain == 'mnist': return tv.datasets.mnist.MNIST(args.data_dir_path, train=False, download=True, transform = grayscale_test_transform)
        if args.domain == 'usps': return tv.datasets.usps.USPS(args.data_dir_path, train=False, download=True, transform = grayscale_test_transform)
        if args.domain == 'svhn': return tv.datasets.svhn.SVHN(args.data_dir_path, split='test', download=True, transform = test_transform)

