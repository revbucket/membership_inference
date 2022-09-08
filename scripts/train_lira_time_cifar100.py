""" Script to train Cifar100 models for computing El2N
    (stealing from https://github.com/MadryLab/datamodels/blob/main/examples/cifar10/train_cifar.py )


Copying ideas from Jon:
- train a bunch of models and collect at even epochs:
    {modelseed, exid, epoch, margin, member}
- later evaluate these and show how lira score changes over time
"""

from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, Identity
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
from pymongo import MongoClient





Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.1),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=20),
    epochs=Param(int, 'How many epochs to run', default=30),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=4),
    gpu=Param(int, 'Which GPU to use', default=0),
    round_size=Param(int, 'How many epochs to train between evaluations', default=2),
    num_rounds=Param(int, 'How many rounds to run', default=15)
)


Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training',
        default='/home/mgj528/datasets/ffcv/cifar100_train'), # REWRITE HARDCODE
    val_dataset=Param(str, '.dat file to use for validation',
        default='/home/mgj528/datasets/ffcv/cifar100_test'), # REWRITE HARDCODE
    mongo_db=Param(str, 'database name for the mongodb',
                   default='lira_time'),
    mongo_coll=Param(str, 'collection name for the mongodb',
                     default='resnet18_cifar100')
)


# ===================================================================================
# =           Creating FFCV Dataloaders + Model                                     =
# ===================================================================================

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('training.gpu')
def make_dataloaders(indices, train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, gpu=None):
    index_set = set(indices)
    anti_indices = [_ for _ in range(50 * 1000) if _ not in index_set]
    paths = {
        'train': train_dataset,
        'test': val_dataset,
        'train_members': train_dataset, # evaluating the training dataset
        'train_nonmembers': train_dataset
    }
    device = 'cuda:%s' % gpu
    start_time = time.time()
    CIFAR100_MEAN = [129.00247, 123.91845, 112.48435]
    CIFAR100_STD = [68.25503, 65.30334, 70.368256]
    loaders = {}
    for name in ['train', 'test', 'train_members', 'train_nonmembers']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        index_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR100_MEAN))),
                Cutout(4, tuple(map(int, CIFAR100_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        if name in ['train', 'train_members']:
            idx_to_use = indices
        elif name == 'train_nonmembers':
            idx_to_use = anti_indices
        else:
            idx_to_use = None


        loaders[name] = Loader(paths[name], indices=idx_to_use,
                               batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline,
                                          'index': index_pipeline})
    return loaders

@param('training.gpu')
def construct_model(gpu=None):
    num_class = 100
    model = torchvision.models.resnet18(pretrained=False, num_classes=100)

    # Modify Resnets to be more CIFAR-friendly
    # https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/cifar10-baseline.ipynb#scrollTo=96ff098b
    model.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = Identity()
    if gpu is not None:
        model = model.to(gpu)
    return model


# ======================================================================================
# =           Training and Evaluation Blocks                                           =
# ======================================================================================


@param('training.lr')
@param('training.epochs')
@param('training.label_smoothing')
@param('training.momentum')
@param('training.weight_decay')
@param('training.lr_peak_epoch')
def setup_train(model, loaders, lr=None, epochs=None, label_smoothing=None, momentum=None,
                weight_decay=None, lr_peak_epoch=None):

    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)
    return opt, scheduler, loss_fn


def resume_train(model, loaders, opt, scheduler, loss_fn, num_epochs):
    model = model.train()
    scaler = GradScaler()
    for _ in tqdm(range(num_epochs)):
        for ims, labs, idxs in loaders['train']:
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()




def evaluate(model, loaders, base_dict):
    """ Here we want the margins for members/nonmembers as indices

    ARGS:
        model + loader are standard
        base_dict: {model_seed, epoch}
        i.e., returns a list of 50,000 docs each of the form
        {base_dict,
         exid: int,
         member: bool,
         margin: float}
    """
    model.eval()
    count = 0
    correct = 0
    output_docs = []
    with ch.no_grad():
        for member, key in [(True, 'train_members'), (False, 'train_nonmembers')]:
            for ims, y, idxs in tqdm(loaders[key]):
                with autocast():
                    bs = y.numel()
                    logits = model(ims)

                    correct_logits = logits[ch.arange(bs), y].clone()
                    logits[ch.arange(bs), y] -= 1000.0
                    next_classes = logits.argmax(1)
                    runnnerup_logits = logits[ch.arange(bs), next_classes].clone()
                    margin = correct_logits - runnnerup_logits


                    for i, m in zip(idxs, margin):
                        new_doc = {'exid': i.item(), 'member': member, 'margin': m.item()}
                        new_doc.update(base_dict)
                        output_docs.append(new_doc)

    return output_docs


@param('training.round_size')
@param('training.num_rounds')
@param('data.mongo_db')
@param('data.mongo_coll')
def main(index, round_size=None, num_rounds=None, mongo_db=None, mongo_coll=None):

    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()


    coll = MongoClient()[mongo_db][mongo_coll]

    indices = list(range(50 * 1000))
    np.random.seed(index)
    np.random.shuffle(indices)
    member_indices = indices[:(25 * 1000)]

    loaders = make_dataloaders(member_indices)
    model = construct_model()

    opt, scheduler, loss_fn = setup_train(model, loaders)
    base_dict = {'epoch': 0, 'model_seed': index}
    coll.insert_many(evaluate(model, loaders, base_dict))

    for round_num in range(num_rounds):
        resume_train(model, loaders, opt, scheduler, loss_fn, round_size)
        base_dict = {'epoch': (round_num + 1) * round_size, 'model_seed': index}
        coll.insert_many(evaluate(model, loaders, base_dict))




