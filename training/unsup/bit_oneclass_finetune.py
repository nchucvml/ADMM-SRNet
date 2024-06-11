import time

import numpy as np

import torch
import torch.optim

from models import timm_wrapper
from utils.utils import AverageMeter

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def bit_mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b

def bit_mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None, simclr_aug_no_crop=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()
        
        batch_size = images.size(0)
        x = images.to(device)
        # unsupervised one class problem has only one target class 
        y = torch.ones_like(labels).unsqueeze(1).float().to(device)

        ### Update learning-rate, including stop training if over. ###
        step = epoch - 1 + n / len(loader)
        lr = timm_wrapper.get_bit_lr(step, len(loader.dataset), P.bit_base_lr)
        if lr is None:
            break
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        ### Process ###
        if P.bit_mixup > 0.0:
            x, y_a, y_b = bit_mixup_data(x, y, P.bit_mixup_l)

        logits = model(x)
        if P.bit_mixup > 0.0:
          loss_cls = bit_mixup_criterion(criterion, logits, y_a, y_b, P.bit_mixup_l)
        else:
          loss_cls = criterion(logits, y)
        loss_cls_num = float(loss_cls.data.cpu().numpy())

        ### Update params ###
        loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        
        P.bit_mixup_l = np.random.beta(P.bit_mixup, P.bit_mixup) if P.bit_mixup > 0 else 1

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_cls_num, batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value))

        check = time.time()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] ' %
         (batch_time.average, data_time.average,
          losses['cls'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)