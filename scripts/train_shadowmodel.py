#!/usr/bin/env python

""" Script to train shadow models.
    Takes in arguments through command line

"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from functools import partial
import sys
import os
import argparse
import random
import wandb

import dataloaders
import shadow_models as sm




# ===================================================================
# =           ARGUMENT PARSING                                      =
# ===================================================================
parser = argparse.ArgumentParser(description="PyTorch-Lightning ShadowModel Training")

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

# Naming and saving conventions
parser.add_argument('project', metavar='PROJECT', help='project name used for logging')
parser.add_argument('expname', metavar='NAME', help='name to be saved')

# Dataset details
parser.add_argument('--dataset', metavar='DATASET', help='which dataset to use') # cifar10 or cifar100
parser.add_argument('--dataseed', metavar='DATASEED', help='which dataset seed to use')
parser.add_argument('--frac', metavar='FRAC', help='fraction of dataset to use')
parser.add_argument('--force_include', metavar='FORCE_INCLUDE', help='indexes to forcefully include', default=None)
parser.add_argument('--force_exclude', metavar='FORCE_EXCLUDE', help='indexes to forcefully exclude', default=None)


# Training details
parser.add_argument('--model', metavar='MODEL', type=str, help='which model to use', default='resnet18')
parser.add_argument('--modelseed', metavar='MODELSEED', type=int, help='seed for initializing model', default=None)
parser.add_argument('--gpus', metavar='GPUS', help='how many gpus to use', default=1)
parser.add_argument('--batch', metavar='BATCH', help='batch size', default=256)
parser.add_argument('--workers', metavar='WORKERS', help='number of workers to use', default=os.cpu_count() / 2)
parser.add_argument('--epochs', metavar='EPOCHS', help='number of epochs to train for', default=100)
parser.add_argument('--save_epochs', metavar='SAVEEPOCHS', help='which epochs to save after', default=[50, 100])


# ===================================================================
# =           MAIN TRAINING BLOCK                                   =
# ===================================================================

def main():
    args = parser.parse_args()
    config_dict = {'project': args.project,
                   'expname': args.expname,
                   'force_include': args.force_include,
                   'force_exclude': args.force_exclude,
                   'model': args.model,
                   'modelseed': args.modelseed,
                   'dataset': args.dataset,
                   'dataseed': args.dataseed}
    print("ARGS", args)
    # Step 1: Set up model
    base_model = sm.load_resnet(args.model, 10, args.modelseed)

    shadow_model = sm.ShadowModel(base_model)


    if args.modelseed is not None:
        torch.manual_seed(random.randrange(0, 2 ** 31))

    # Step 2: Set up dataset
    datamodule = dataloaders.CifarSubset(frac=args.frac, seed=args.dataseed,
                                         dataset_dir=args.data,
                                         which_cifar=args.dataset,
                                         xform_args={'pad': 2, 'mirror': True, 'normalize': True},
                                         force_include=args.force_include,
                                         force_exclude=args.force_exclude)
    datamodule.setup()
    train_loader = datamodule.train_dataloader(batch_size=args.batch, num_workers=args.workers)
    val_loader = datamodule.val_dataloader(batch_size=args.batch, num_workers=args.workers)


    checkpoint_path = os.path.join(os.getcwd(), 'shadows', args.project, args.expname,
                                   '%s_dataseed%s_modelseed%s' % (args.model, args.dataseed, args.modelseed))
    # Step 4: Setup trainer and train
    logger = WandbLogger(project=args.project, name=args.expname)
    logger.experiment.config.update(config_dict)

    trainer_kwargs = {'accelerator': 'gpu',
                      'devices': args.gpus,
                      'max_epochs': args.epochs,
                      'progress_bar_refresh_rate': 10,
                      'default_root_dir': checkpoint_path,
                      'logger': logger}
    if args.gpus > 1:
        trainer_kwargs['strategy'] = 'ddp'
    trainer = pl.Trainer(**trainer_kwargs)


    trainer.fit(shadow_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
