import torch, os
import numpy as np 
import torchvision as tv

def data_sel(args, train=True):
    if train ==True: return train_selector(args)
    else: return test_selector(args)


def train_selector(args):
    cifar_path = os.path.join(self.root, 'cifar-10-batches-py')
    if not os.path.isdir(cifar_path):
        train_ds = tv.datasets.CIFAR10(root = cifar_path, download = True)
    else:
        train_ds = tv.datasets.CIFAR10(root = cifar_path, download = False)
    
    return train_ds

def test_selector(args):
    cifar_path = os.path.join(self.root, 'cifar-10-batches-py')
    corr_path = os.path.join(self.root, 'CIFAR-10-C')
    if args.domain == 0:
        test_ds = torchvision.datasets.CIFAR10(root = cifar_path, train=False)
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

        all_corr_x = []
        corr = args.domain
        for i in range(19):
            if i = 0:
                all_corr_dset = torch.utils.data.TensorDataset(all_corr[i][ (corr-1)*10000 : corr*10000], labels[(corr-1)*10000 : corr*10000] )
            else: 
                curr_corr_dset = torch.utils.data.TensorDataset(all_corr[i][ (corr-1)*10000 : corr*10000], labels[(corr-1)*10000 : corr*10000] )
                all_corr_dset = torch.utils.data.ConcatDataset(all_corr_dset, curr_corr_dset)
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
            




