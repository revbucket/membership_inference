""" Script to train Cifar100 models for computing El2N
    (stealing from https://github.com/MadryLab/datamodels/blob/main/examples/cifar10/train_cifar.py )
"""

from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

# ======================================================================
# =           Setting Hyperparameters                                  =
# ======================================================================

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.1),
    epochs=Param(int, 'Number of epochs to run for', default=201),
    warm=Param(int, 'Number of epochs to warmup for', default=1),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    milestones=Param(str, 'Epochs at which to reduce LR: Semicolon separated string', default='60;120;160'),
    num_workers=Param(int, 'The number of workers', default=4),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    gpu=Param(int, 'Which GPU to use', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training',
        default='/home/mgj528/datasets/ffcv/cifar100_train'), # REWRITE HARDCODE
    val_dataset=Param(str, '.dat file to use for validation',
        default='/home/mgj528/datasets/ffcv/cifar100_test'), # REWRITE HARDCODE
)




# ===================================================================================
# =           Creating FFCV Dataloaders + Model                                     =
# ===================================================================================

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('training.gpu')
def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, gpu=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }
    device = 'cuda:%s' % gpu
    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        index_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            #torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL
        loaders[name] = Loader(paths[name], indices=None,
                               batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline,
                                          'index': index_pipeline})
    return loaders

@param('training.gpu')
def construct_model(gpu=None):
    num_class = 100
    model = torchvision.models.resnet18(pretrained=False, num_classes=100)
    if gpu is not None:
        model = model.to(gpu)
    return model


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


# ======================================================================================
# =           Training and Evaluation Blocks                                           =
# ======================================================================================


@param('training.lr')
@param('training.warm')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.milestones')
def train(model, loaders, lr=None, warm=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, milestones=None):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    milestones = [int(_) for _ in milestones.split(';')]
    train_scheduler = lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.2)
    # Use LR schedule from https://github.com/weiaicunzai/pytorch-cifar100
    # Cyclic LR with single triangle
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in tqdm(range(epochs)):
        for ims, labs, idxs in loaders['train']:
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        train_scheduler.step(epoch)




@param('training.lr_tta')
def evaluate(model, loaders, lr_tta=False):
    model.eval()
    count = 0
    correct = 0
    with ch.no_grad():
        all_margins = []
        for ims, labs, idxs in tqdm(loaders['test']):
            count += labs.numel()

            with autocast():
                out = model(ims)
                if lr_tta:
                    out += model(ch.fliplr(ims))
                    out /= 2
                correct += (out.max(dim=1)[1] == labs).sum().item()
                class_logits = out[ch.arange(out.shape[0]), labs].clone()
                out[ch.arange(out.shape[0]), labs] = -1000
                next_classes = out.argmax(1)
                class_logits -= out[ch.arange(out.shape[0]), next_classes]
                all_margins.append(class_logits.cpu())
        all_margins = ch.cat(all_margins)

        print('Average margin:', all_margins.mean())
        print("Top1 Accuracy", float(correct) / count)
        return (all_margins.numpy(), float(correct) / count)

def main(index, logdir):
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders = make_dataloaders()
    model = construct_model()
    train(model, loaders)
    margins, acc = evaluate(model, loaders)
    print(margins.shape)
    return {
        'acc': acc,
        'margins': margins
    }