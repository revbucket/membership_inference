""" Script to train Cifar100 models for membership inference

"""


from argparse import ArgumentParser
from typing import List
import time
import os
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, Identity
import torch.nn.functional as F
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

# import pruning_metrics as pm

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
    gpu=Param(int, 'Which GPU to use', default=3), # Final A100 on DGX
    save_path=Param(str, 'location of path to save models',
                    default='/home/mgj528/grad/membership_inference/save/cifar100_mi/resnet18'),
    base_name=Param(str, 'prefix of model names, like <base_name>_seed', default='seed'),
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training',
        default='/home/mgj528/datasets/ffcv/cifar100_train'), # REWRITE HARDCODE
    val_dataset=Param(str, '.dat file to use for validation',
        default='/home/mgj528/datasets/ffcv/cifar100_test'), # REWRITE HARDCODE
    mongo_db=Param(str, 'database name for the mongodb', default='lira'),
    mongo_coll=Param(str, 'collection name for the mongodb', default='resnet18_cifar100')
)




@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('training.gpu')
def make_dataloaders(indices, train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, gpu=None):
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
        loaders[name] = Loader(paths[name], indices=indices,
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



def evaluate_privacy(index, model, loaders, anti_loaders):
    """ Creates a list of mongo documents for all train/test images:
        each doc looks a little like:
        {seed: which seed of data to use,
         exid: example_id,
         train: boolean if in training or test set,
         member: boolean if in training set's membership,
         xentropy: crossEntropy score,
         margin: correct vs incorrect margin,
         one_v_all: correct logit minus logsumexp of other logits
         }
    ARGS:
        index: which seed to use
        model: loaded/trained model
        loaders: result of make_dataloaders (with the training indices!)
        anti_loaders: result of make_dataloaders with the inverse trainingindices
    RETURNS:
        list of docs (to be inserted into mongo)
    """

    model = model.eval()
    all_docs = []
    for (loader_dict, member) in [(loaders, True), (anti_loaders, False)]:
        for k in ['eval_train']:
            with ch.no_grad():
                for x, y, idxs in loader_dict[k]:
                    with autocast():
                        bs = y.numel()
                        logits = model(x)
                        xentropy = F.cross_entropy(logits, y, reduction='none')
                        correct_logits = logits[ch.arange(bs), y].clone()
                        logits[ch.arange(bs), y] -= 1000.0
                        next_classes = logits.argmax(1)
                        runnnerup_logits = logits[ch.arange(bs), next_classes].clone()
                        margin = correct_logits - runnnerup_logits
                        one_v_all = ch.log(ch.exp(logits).sum(dim=1))
                    data = ch.stack([idxs, xentropy, margin, one_v_all]).T
                    for exid, xent, marg, ova in data:
                        new_doc = {'exid': int(exid.item()),
                                   'xentropy': xent.item(),
                                   'margin': marg.item(),
                                   'one_v_all': ova.item(),
                                   'seed': index,
                                   'train': True,
                                   'member': member}

                        all_docs.append(new_doc)

    return all_docs










@param('training.epochs')
@param('training.save_path')
@param('training.base_name')
@param('data.mongo_db')
@param('data.mongo_coll')
def main(index, epochs=None, save_path=None, base_name=None,
         mongo_db=None, mongo_coll=None):
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
    pro_indices = indices[:(25 * 1000)]
    anti_indices = indices[(25 * 1000):]

    loaders = make_dataloaders(pro_indices)
    anti_loaders = make_dataloaders(anti_indices)
    model = construct_model()

    # Train 50 epochs
    opt, scheduler, loss_fn = setup_train(model, loaders)
    resume_train(model, loaders, opt, scheduler, loss_fn, epochs)


    # And evaluate afterwards
    mongo_docs = evaluate_privacy(index, model, loaders, anti_loaders)
    coll.insert_many(mongo_docs)

    # And then save the model weights
    model = model.cpu()
    if not(os.path.isdir(save_path)):
        os.makedirs(save_path, exist_ok=True)
        
    model_path = os.path.join(save_path, base_name + '_%04d' % index)
    ch.save(model.state_dict(), model_path)
