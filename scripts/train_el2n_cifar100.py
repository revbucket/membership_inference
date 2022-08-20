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

import pruning_metrics as pm

# ======================================================================
# =           Setting Hyperparameters                                  =
# ======================================================================

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.1),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=20),
    epochs=Param(int, 'Number of epochs to run for', default=50),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=4),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    gpu=Param(int, 'Which GPU to use', default=0),
    round_size=Param(int, 'How many epochs to train between evaluations', default=2),
    num_rounds=Param(int, 'How many rounds to run', default=10)

)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training',
        default='/home/mgj528/datasets/ffcv/cifar100_train'), # REWRITE HARDCODE
    val_dataset=Param(str, '.dat file to use for validation',
        default='/home/mgj528/datasets/ffcv/cifar100_test'), # REWRITE HARDCODE
    mongo_db=Param(str, 'database name for the mongodb',
                   default='datadiet'),
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
def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, gpu=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset,
        'eval_train': train_dataset, # evaluating the training dataset
    }
    device = 'cuda:%s' % gpu
    start_time = time.time()
    CIFAR100_MEAN = [129.00247, 123.91845, 112.48435]
    CIFAR100_STD = [68.25503, 65.30334, 70.368256]
    loaders = {}
    for name in ['train', 'test', 'eval_train']:
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


def evaluate_pruning(model, model_id, loaders, epoch):
    """ Builds a list of documents to insert into mongo
        for the pruning metrics (el2n, grand)

        Each document looks like:
        {model_id: int - unique indentifier for this model
         epoch: int - which epoch this model was at
         train: bool - whether this example was a training example
         ex_id: int - example number
         el2n: float - el2n score for this example
         grand: float - grand score for this example}
    """
    model = model.eval()
    output = []

    for k in ('eval_train', 'test'):
        for batch in tqdm(loaders[k]):
            el2n_batch = pm.el2n_minibatch(model, batch, 100).cpu().data
            grand_batch = pm.grand_minibatch(model, batch)

            output.extend([{'model_id': model_id,
                            'epoch': epoch,
                            'train': (k == 'train'),
                            'el2n': el2n_batch[i].item(),
                            'grand': grand_batch[i].item(),
                            'exid': batch[2][i].item()} for i in range(len(batch[2]))])


    return output


@param('data.mongo_db')
@param('data.mongo_coll')
@param('training.round_size')
@param('training.num_rounds')
def main(index, mongo_db=None, mongo_coll=None,
         round_size=None, num_rounds=None):
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()


    coll = MongoClient()[mongo_db][mongo_coll]
    loaders = make_dataloaders()
    model = construct_model()

    # Train 2x10
    opt, scheduler, loss_fn = setup_train(model, loaders)

    for round_num in range(1, num_rounds + 1):
        resume_train(model, loaders, opt, scheduler, loss_fn, round_size)
        data = evaluate_pruning(model, index, loaders, round_size * round_num)
        coll.insert_many(data)




