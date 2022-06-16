""" Handles the training of shadow models """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import glob
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics


full_path = os.path.realpath(__file__)



def load_resnet(resnet_name, num_classes, model_seed=None):
    if model_seed is not None:
        torch.manual_seed(model_seed)
    model = eval('torchvision.models.%s()' % resnet_name)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model




# ======================================================================
# =           Shadow Module Lightning Module                           =
# ======================================================================

class ShadowModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)
        return {'loss': loss, 'preds': y_pred, 'target': y}

    def training_step_end(self, outs):
        self.log('train/loss', outs['loss'])
        self.train_acc(outs['preds'], outs['target'])
        self.log('train/acc', self.train_acc)

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)
        return {'loss': loss, 'preds': y_pred, 'target': y}

    def validation_step_end(self, outs):
        self.val_acc(outs['preds'], outs['target'])
        self.log('val/loss', outs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True, sync_dist=True)

    ############# Optimizer configuration ########################

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1,
                              weight_decay = 0.0005, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160],
                                                   gamma=0.2)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)

