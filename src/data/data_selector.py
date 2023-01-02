from .data_utils import train_selector, test_selector
from .load_attack_data import load_attacked

def data_sel(args, train=True, only_aug=False, adv=False):
    if adv:
        # Load adversarial training based datasets
        return load_attacked(args, train=train, only_adv=only_aug)

    if train ==True: return train_selector(args, only_aug=only_aug)
    else: return test_selector(args)




