'''
    ACVC single domian generalization training approach

    1) augment dataset with visual corruptions
    2) Include an attention consistency loss in the loss function during training
'''

from .single_density_sampling import SingleDensitySampleTrainer
from ..tools.tools import AverageMeter, accuracy_topk, print_log, get_ds_range
from ..models.model_tools import get_cam_pred

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch
import numpy as np
from tqdm import tqdm
from imagecorruptions import corrupt, get_corruption_names
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime

class ACVCTrainer(SingleDensitySampleTrainer):
    '''
    Use augmented training set to learn a density distribution
    apply dist_transform to joint likelihood of x and x_aug to obtain weights for training
    '''
    def __init__(self, aug_ds, device, model, optimizer, criterion, scheduler, kernel='gaussian', bandwidth=1, kde_frac=1.0, df=False):
        SingleDensitySampleTrainer.__init__(self, self.flatten_aug_ds(aug_ds), device, model, optimizer, criterion, scheduler, kernel, bandwidth, kde_frac, df=df)

    @staticmethod
    def aug_ds(ds):
        '''
        input: ds (x,y)
        return: aug_ds (x, augmix1, augmix2, y)
        
        order is maintained
        '''
        transform = ACVCGenerator.transform
        xs = []
        x_augs = []
        ys = []
        print('generating VC aug ds')
        small, big = get_ds_range(ds)
        small = small.unsqueeze(dim=-1).unsqueeze(dim=-1)
        big = big.unsqueeze(dim=-1).unsqueeze(dim=-1)
        r = big-small
        for i in tqdm(range(len(ds))):
            x, y = ds[i]
            x_scaled  = (((x - small)/r) * 255).to(torch.uint8) # values between 0-255 for transformation
            x_augs.append(((transform(x_scaled)/255)*r) + small)
            ys.append(y)
            xs.append(x)
        xs = torch.stack(xs, dim=0)
        x_augs = torch.stack(x_augs, dim=0)
        ys = torch.LongTensor(ys)
        return TensorDataset(xs, x_augs, ys)

    @staticmethod
    def flatten_aug_ds(aug_ds):
        '''
        Input: each sample in aug_ds is (x, x_aug, y)
        Output: each above sample becomes 2 samples: (x, y), (x_aug, y)
        '''

        xs = []
        ys = []
        for i in range(len(aug_ds)):
            x, x_aug, y = aug_ds[i]
            xs += [x, x_aug]
            ys += [y, y]
        xs = torch.stack(xs, dim=0)
        ys = torch.LongTensor(ys)
        return TensorDataset(xs, ys)

    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
        '''
        Run one train epoch for acvc
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to train mode
        model.train()

        for i, (x, x_aug, y) in enumerate(train_loader):
            x = x.to(device)
            x_aug = x_aug.to(device)
            y = y.to(device)

            # Forward pass
            logits_clean = model(x)
            logits_aug = model(x_aug)

            # classification loss
            loss_classification = criterion(logits_clean, y) + criterion(logits_aug, y)

            # 'contrastive loss' - in this case the ACVC loss
            c_clean = get_cam_pred(model, x)
            c_aug = get_cam_pred(model, x_aug)

            acvc_loss_calculator = AttentionConsistency()
            loss_acvc = acvc_loss_calculator(c_clean, [c_aug], y)

            loss = loss_classification + loss_acvc

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
    

    def prep_weighted_dl(self, aug_ds, dist_transform='unity', gamma=1.0, bs=64, transform_args=None):
        '''
        Creates dl with samples drawn to create each batch randomly using weighting as defined by dist_transform on likelihoods
        '''
        print("Getting weights", datetime.now())
        tw = self.get_weights(self.train_dist_model, self.flatten_aug_ds(aug_ds)) # s(x)
        
        # unflatten weights and add: p(x)+p(aug)
        tw_splits = [split.squeeze() for split in np.vsplit(np.expand_dims(tw, axis=1), len(aug_ds))]
        train_weights = np.asarray([split[0]+split[1] for split in tw_splits])
        
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





class ACVCGenerator:
    """
    Generate Visually Corrupted images as per ACVC

    Code sourced from:
    https://github.com/ExplainableML/ACVC/blob/main/preprocessing/image/ACVCGenerator.py
    """

    @classmethod
    def transform(cls, x):
        '''
            Return visually corrupted x
            x: Torch.Tensor[3 x w x h] with pixel values in the range 0-255
        '''

        if x.shape[-1] < 32:
            x = F.interpolate(x.unsqueeze(dim=0), size=[32,32]).squeeze()
        
        x = torch.transpose(x,0,2)
        out = torch.from_numpy(cls.acvc(x.numpy()))
        out = out.type(torch.FloatTensor)
        out = torch.transpose(out,0,2)
        return F.interpolate(out.unsqueeze(dim=0), size=[16,16]).squeeze()


    @staticmethod
    def get_severity():
        return np.random.randint(1, 6)

    @staticmethod
    def draw_cicle(shape, diamiter):
        """
        Input:
        shape    : tuple (height, width)
        diameter : scalar

        Output:
        np.array of shape  that says True within a circle with diamiter =  around center
        """
        assert len(shape) == 2
        TF = np.zeros(shape, dtype="bool")
        center = np.array(TF.shape) / 2.0

        for iy in range(shape[0]):
            for ix in range(shape[1]):
                TF[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diamiter ** 2
        return TF

    @staticmethod
    def filter_circle(TFcircle, fft_img_channel):
        temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
        temp[TFcircle] = fft_img_channel[TFcircle]
        return temp

    @staticmethod
    def inv_FFT_all_channel(fft_img):
        img_reco = []
        for ichannel in range(fft_img.shape[2]):
            img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, ichannel])))
        img_reco = np.array(img_reco)
        img_reco = np.transpose(img_reco, (1, 2, 0))
        return img_reco

    @classmethod
    def high_pass_filter(cls, x, severity):
        x = x.astype("float32") / 255.
        c = [.01, .02, .03, .04, .05][severity - 1]

        d = int(c * x.shape[0])
        TFcircle = cls.draw_cicle(shape=x.shape[:2], diamiter=d)
        TFcircle = ~TFcircle

        fft_img = np.zeros_like(x, dtype=complex)
        for ichannel in range(fft_img.shape[2]):
            fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(x[:, :, ichannel]))

        # For each channel, pass filter
        fft_img_filtered = []
        for ichannel in range(fft_img.shape[2]):
            fft_img_channel = fft_img[:, :, ichannel]
            temp = cls.filter_circle(TFcircle, fft_img_channel)
            fft_img_filtered.append(temp)
        fft_img_filtered = np.array(fft_img_filtered)
        fft_img_filtered = np.transpose(fft_img_filtered, (1, 2, 0))
        x = np.clip(np.abs(cls.inv_FFT_all_channel(fft_img_filtered)), a_min=0, a_max=1)

        # x = PILImage.fromarray((x * 255.).astype("uint8"))
        x = x*255
        return x

    @classmethod
    def constant_amplitude(cls, x, severity):
        """
        A visual corruption based on amplitude information of a Fourier-transformed image

        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.05, .1, .15, .2, .25][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Amplitude replacement
        beta = 1.0 - c
        x_abs = np.ones_like(x_abs) * max(0, beta)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        # x = PILImage.fromarray((x * 255.).astype("uint8"))
        x = x*255
        return x

    @classmethod
    def phase_scaling(cls, x, severity):
        """
        A visual corruption based on phase information of a Fourier-transformed image

        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.1, .2, .3, .4, .5][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Phase scaling
        alpha = 1.0 - c
        x_pha = x_pha * max(0, alpha)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        # x = PILImage.fromarray((x * 255.).astype("uint8"))
        x = x*255
        return x

    @classmethod
    def apply_corruption(cls, x, corruption_name):
        severity = cls.get_severity()

        custom_corruptions = {"high_pass_filter": cls.high_pass_filter,
                              "constant_amplitude": cls.constant_amplitude,
                              "phase_scaling": cls.phase_scaling}

        if corruption_name in get_corruption_names('all'):
            x = corrupt(x, corruption_name=corruption_name, severity=severity)
            # x = PILImage.fromarray(x)

        elif corruption_name in custom_corruptions:
            x = custom_corruptions[corruption_name](x, severity=severity)

        else:
            assert True, "%s is not a supported corruption!" % corruption_name

        return x

    @classmethod
    def acvc(cls, x):
        i = np.random.randint(0, 22)
        corruption_func = {0: "fog",
                           1: "snow",
                           2: "frost",
                           3: "spatter",
                           4: "zoom_blur",
                           5: "defocus_blur",
                           6: "glass_blur",
                           7: "gaussian_blur",
                           8: "motion_blur",
                           9: "speckle_noise",
                           10: "shot_noise",
                           11: "impulse_noise",
                           12: "gaussian_noise",
                           13: "jpeg_compression",
                           14: "pixelate",
                           15: "elastic_transform",
                           16: "brightness",
                           17: "saturate",
                           18: "contrast",
                           19: "high_pass_filter",
                           20: "constant_amplitude",
                           21: "phase_scaling"}
        return cls.apply_corruption(x, corruption_func[i])


class AttentionConsistency(nn.Module):
    def __init__(self, lambd=6e-2, T=1.0):
        super().__init__()
        self.name = "AttentionConsistency"
        self.T = T
        self.lambd = lambd

    def CAM_neg(self, c):
        result = c.reshape(c.size(0), c.size(1), -1)
        result = -nn.functional.log_softmax(result / self.T, dim=2) / result.size(2)
        result = result.sum(2)

        return result

    def CAM_pos(self, c):
        result = c.reshape(c.size(0), c.size(1), -1)
        result = nn.functional.softmax(result / self.T, dim=2)

        return result

    def forward(self, c, ci_list, y):
        """
        CAM (batch_size, num_classes, feature_map.shpae[0], feature_map.shpae[1]) based loss

        Arguments:
            :param c: (Torch.tensor) clean image's CAM
            :param ci_list: (Torch.tensor) list of augmented image's CAMs
            :param y: (Torch.tensor) ground truth labels
        :return:
        """
        c1 = c.clone()
        c1 = Variable(c1)
        c0 = self.CAM_neg(c)

        # Top-k negative classes
        c1 = c1.sum(2).sum(2)
        index = torch.zeros(c1.size())
        c1[range(c0.size(0)), y] = - float("Inf")
        topk_ind = torch.topk(c1, 3, dim=1)[1]
        index[torch.tensor(range(c1.size(0))).unsqueeze(1), topk_ind] = 1
        index = index > 0.5

        # Negative CAM loss
        neg_loss = c0[index].sum() / c0.size(0)
        for ci in ci_list:
            ci = self.CAM_neg(ci)
            neg_loss += ci[index].sum() / ci.size(0)
        neg_loss /= len(ci_list) + 1

        # Positive CAM loss
        index = torch.zeros(c1.size())
        true_ind = [[i] for i in y]
        index[torch.tensor(range(c1.size(0))).unsqueeze(1), true_ind] = 1
        index = index > 0.5
        p0 = self.CAM_pos(c)[index]
        pi_list = [self.CAM_pos(ci)[index] for ci in ci_list]

        # Middle ground for Jensen-Shannon divergence
        p_count = 1 + len(pi_list)

        p_mixture = p0.detach().clone()
        for pi in pi_list:
            p_mixture += pi
        p_mixture = torch.clamp(p_mixture / p_count, 1e-7, 1).log()

        pos_loss = nn.functional.kl_div(p_mixture, p0, reduction='batchmean')
        for pi in pi_list:
            pos_loss += nn.functional.kl_div(p_mixture, pi, reduction='batchmean')
        pos_loss /= p_count

        loss = pos_loss + neg_loss
        return self.lambd * loss
    