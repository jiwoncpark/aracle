# -*- coding: utf-8 -*-
"""Training the Bayesian neural network (BNN).
This script trains the BNN according to the config specifications.
Example
-------
To run this script, pass in the path to the user-defined training config file as the argument::
    
    $ train h0rton/example_user_config.py
"""

import os, sys
import random
import argparse
import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm
# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
# Aracle
from aracle.configs import AracleConfig
from aracle.data import ResNetDataset

def seed_everything(global_seed):
    """Seed everything for reproducibility
    global_seed : int
        seed for `np.random`, `random`, and relevant `torch` backends
    """
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    from aracle.user_cfg import cfg
    cfg = AracleConfig(cfg)
    seed_everything(cfg.global_seed)

    # Define training data and loader
    train_data = ResNetDataset(cfg.data.train_dir, cfg.data.t_offset, cfg.data.normalize_X, cfg.data.X_mean, cfg.data.X_std)
    train_loader = DataLoader(train_data, batch_size=cfg.optim.batch_size, shuffle=True)
    n_train = train_data.n_data

    # Define val data and loader
    val_data = ResNetDataset(cfg.data.val_dir, cfg.data.t_offset, cfg.data.normalize_X, cfg.data.X_mean, cfg.data.X_std)
    val_loader = DataLoader(val_data, batch_size=cfg.optim.batch_size, shuffle=True)
    n_val = val_data.n_data

    # Instantiate loss function
    loss_fn = nn.CrossEntropyLoss()
    # Instantiate model
    net = torchvision.models.segmentation.fcn_resnet101(pretrained=cfg.model.pretrained, num_classes=2)
    net.to(cfg.device)

    # Instantiate optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.optim.learning_rate, amsgrad=True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.optim.lr_scheduler.milestones, gamma=cfg.optim.lr_scheduler.gamma)
    logger = SummaryWriter()

    if not os.path.exists(cfg.log.checkpoint_dir):
        os.mkdir(cfg.log.checkpoint_dir)
        #net = torch.load('./saved_model/resnet18.mdl')
        #print('loaded mdl!')

    progress = tqdm(range(cfg.optim.n_epochs))
    for epoch in progress:
        net.train()
        total_loss = 0.0

        for batch_idx, (X_, Y_) in enumerate(train_loader):
            #X = Variable(torch.Tensor(X_)).to(cfg.device)
            #Y = Variable(torch.Tensor(Y_)).to(cfg.device)
            X = X_.to(cfg.device)
            Y = Y_.to(cfg.device)
            batch_size = X.shape[0]

            pred = net(X)
            loss = loss_fn(pred, Y)
            total_loss += loss.item()*batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        with torch.no_grad():
            net.eval()
            total_val_loss = 0.0

            for batch_idx, (X_, Y_) in enumerate(val_loader):
                X = X_.to(cfg.device)
                Y = Y_.to(cfg.device)
                batch_size = X.shape[0]

                pred = net(X)
                loss = loss_fn(pred, Y)
                total_val_loss += loss.item()*batch_size

            epoch_avg_train_loss = total_loss/n_train
            epoch_avg_val_loss = total_val_loss/n_val

            tqdm.write("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, epoch_avg_train_loss))
            tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, epoch_avg_val_loss))
            
            if (epoch + 1)%(cfg.log.logging_interval) == 0:
                # Log train and val losses
                logger.add_scalars('metrics/loss',
                                   {'train': epoch_avg_train_loss, 'val': epoch_avg_val_loss},
                                   epoch)

                # Log histograms of named parameters
                for param_name, param in net.named_parameters():
                    logger.add_histogram(param_name, param.clone().cpu().data.numpy(), epoch)

            if (epoch + 1)%(cfg.log.checkpoint_interval) == 0:
                time_stamp = str(datetime.date.today())
                torch.save(net, os.path.join(cfg.log.checkpoint_dir, 'resnet18_{:s}.mdl'.format(time_stamp)))

    logger.close()

if __name__ == '__main__':
    main()