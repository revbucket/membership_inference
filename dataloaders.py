""" Sets up data loaders for loading subsets of datasets """

import torch
import torchvision.datasets as datasets
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import numpy as np
import pytorch_lightning as pl
import hashlib

DEFAULT_DATASET_DIR = os.path.expanduser('~/datasets')

CIFAR10_MEAN = (0.4914, 0.4822,     0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


# =================================================
# =           General Subset Utilities            =
# =================================================
"""
Strategy for subsets:
- Get a seed (which is an integer)
- Hash the (string(idx) + str(seed))
- Check hexdigest
"""

class IndexSubset(torch.utils.data.Subset):
    def __init__(self, dataset, idxs) -> None:
        self.idxs = idxs
        self.dataset = torch.utils.data.Subset(dataset, idxs)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return (x, y, self.idxs[idx])

    def __len__(self):
        return len(self.dataset)


def check_containment(idx, frac, seed):
    """ Checks if element idx is in subset (defined by seed) and frac
    ARGS:
        idx: (int) - index of element we are checking containment
        frac: (float) - which percentage of subset to KEEP
        seed: (int) - seed to define this 'instantiation' of subset
    RETURNS:
        (bool) - if idx is in subset
    """
    to_hash = (str(int(idx)) + str(seed)).encode('utf-8')
    hashed = int(hashlib.sha256(to_hash).hexdigest()[-8:], 16)
    threshold = int('f' * 8, 16) * frac
    return hashed < threshold


def get_index_subset(base_dataset, frac, seed, complement=False,
                     force_include=None, force_exclude=None):
    """ Creates an indexed subset from a base dataset
    ARGS:
        base_dataset (dataset) - torch.utils.dataset object, like CIFAR10/MNIST
        frac: (float) - fraction of the dataset to keep
        seed: (int) - random seed to use as salt for containment checking
        complement (False) - if True we get the 'opposite' of this subset
    RETURNS:
        - Subset: subset of dataset with only passing idxs
    """

    numel = len(base_dataset)
    idxs = set([_ for _ in range(numel) if check_containment(_, frac, seed)])

    if complement:
        idxs = set([_ for _ in range(numel) if _ not in idxs])


    if force_include is not None:
        if isinstance(force_include, int):
            force_include = [force_include]
        idxs = idxs.union(set(force_include))

    if force_exclude is not None:
        if isinstance(force_exclude, int):
            force_exclude = [force_exclude]
        idxs = idxs.difference(set(force_exclude))
    idxs = sorted(list(idxs))

    return IndexSubset(base_dataset, idxs)




# =================================================================
# =           CIFAR 10/100 Augmentation stuff                     =
# =================================================================

def cifar_augmentation(pad, mirror=False, normalize=True,
                       which_cifar='cifar10'):

    """ Builds augmentations for cifar models
        Order goes like [mirroring, padding, ToTensor, Normalize]
    ARGS:
        pad: (int) - how much to pad
        mirror: (bool) - whether or not to do random horizontal flips
        normalize: (bool) - whether or not to do normalization
        which_cifar: (str) - 'cifar10' or 'cifar100'; affects normalization
    """

    xform_list = []
    if mirror:
        xform_list.append(transforms.RandomHorizontalFlip())
    if pad > 0:
        xform_list.append(transforms.RandomCrop(32, padding=pad))

    xform_list.append(transforms.ToTensor())

    if normalize:
        mean, std = {'cifar10': (CIFAR10_MEAN, CIFAR10_STD),
                     'cifar100': (CIFAR100_MEAN, CIFAR100_STD)}[which_cifar]
        xform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(xform_list)


class CifarSubset(pl.LightningDataModule):
    def __init__(self, frac:float,
                 seed: int=0,
                 dataset_dir: str= DEFAULT_DATASET_DIR,
                 which_cifar: str='cifar10',
                 xform_args: dict=None,
                 force_include=None,
                 force_exclude=None
                 ):
        super().__init__()
        self.frac = frac
        self.seed = seed
        self.dataset_dir = dataset_dir
        self.which_cifar = which_cifar


        if xform_args is None:
            xform_args = {'pad': 0, 'mirror': False,
                          'normalize': False}
        xform_args['which_cifar'] = which_cifar
        self.xform_args = xform_args

        self.force_include = force_include
        self.force_exclude = force_exclude

    def setup(self, stage=None):
        dataset = {'cifar10': datasets.CIFAR10,
                   'cifar100': datasets.CIFAR100}[self.which_cifar]

        # Setup training dataset
        train_xform = cifar_augmentation(**self.xform_args)
        self.full_trainset = dataset(root=self.dataset_dir,
                                     train=True, download=True,
                                     transform=train_xform)
        self.cifar_train = get_index_subset(self.full_trainset,
                                            self.frac, self.seed,
                                            force_include=self.force_include,
                                            force_exclude=self.force_exclude)


        # Setup validation dataset
        val_transform = cifar_augmentation(pad=0, mirror=False,
                                           normalize=self.xform_args['normalize'],
                                           which_cifar=self.which_cifar)

        self.full_valset = self.cifar_val = dataset(root=self.dataset_dir,
                                                    train=False, download=True,
                                                    transform=val_transform)



    def train_dataloader(self, batch_size=128, num_workers=4):
        return DataLoader(self.cifar_train, batch_size=batch_size,
                          shuffle=True, pin_memory=torch.cuda.is_available(),
                          num_workers=num_workers)

    def val_dataloader(self, batch_size=128, num_workers=4):
        return DataLoader(self.cifar_val, batch_size=batch_size,
                          shuffle=False, pin_memory=torch.cuda.is_available(),
                          num_workers=num_workers)








