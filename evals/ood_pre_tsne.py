"""
Compute t-SNE visualization of the learned features 

Curently this can only process one class features of CIFAR
"""

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np

from sklearn import manifold

from tqdm import tqdm

import models.transform_layers as TL
from utils.utils import set_random_seed, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def eval_ood_tsne(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory

    prefix = f'{P.ood_samples}'
    if P.resize_fix:
        prefix += f'_resize_fix_{P.resize_factor}'
    else:
        prefix += f'_resize_range_{P.resize_factor}'

    prefix = os.path.join(base_path, f'feats_{prefix}')

    feature_layer = P.ood_layer[0] # 'penultimate'

    if P.one_class_idx is not None:
        P.target_class = P.one_class_idx
        label_map = None
    else:
        if P.dataset == 'cifar10':
            P.target_class = 'cifar10'
            label_map = {'cifar10': 0, 'svhn': 1, 'lsun_resize': 2, 'imagenet_resize': 3, 'lsun_fix': 4, 'imagenet_fix': 5, 'cifar100': 6, 'interp': 7}
        elif P.dataset == 'imagenet':
            P.target_class = 'imagenet'
            label_map = {'imagenet': 0, 'cub': 1, 'stanford_dogs': 2, 'flowers102': 3, 'places365': 4, 'food_101': 5, 'caltech_256': 6, 'dtd': 7, 'pets': 8}

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': [feature_layer], #P.ood_layer,
    }

    print('Compute train set features')
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)
    feats_train_mean = [f.mean(dim=1) for f in feats_train[feature_layer].chunk(P.K_shift, dim=1)]  # list of (M, d)   
    train_count =  feats_train_mean[0].shape[0]
    label_train = [P.target_class] * train_count # M
    feats_train_np = [f.cpu().detach().numpy() for f in feats_train_mean] # list of (M, d)

    print('Compute test set features')
    feats_id = get_features(P, P.dataset, model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    feats_id_mean = [f.mean(dim=1) for f in feats_id[feature_layer].chunk(P.K_shift, dim=1)]  # list of (N, d)
    label_id = [P.target_class] * feats_id_mean[0].shape[0] # N
    feats_id_np = [f.cpu().detach().numpy() for f in feats_id_mean] # list of (N, d)
    
    feats_ood = dict()
    feats_ood_mean = dict()
    label_ood = dict()
    feats_ood_np = dict()
    for ood_idx, (ood, ood_loader) in enumerate(ood_loaders.items()):
        ood_class = P.ood_dataset[ood_idx]

        if ood == 'interp':
            feats_ood[ood] = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, **kwargs)
        else:
            feats_ood[ood] = get_features(P, ood, model, ood_loader, prefix=prefix, **kwargs)
        
        feats_ood_mean[ood_class] = [f.mean(dim=1) for f in feats_ood[ood][feature_layer].chunk(P.K_shift, dim=1)]  # list of (N, d)
        label_ood[ood_class] = [ood_class] * feats_ood_mean[ood_class][0].shape[0] # N
        feats_ood_np[ood_class] = [f.cpu().detach().numpy() for f in feats_ood_mean[ood_class]] # list of (N, d)
    
    # collect and convert features to the form of tsne input
    feats_all_mean = [[feats_train_np[k], feats_id_np[k]] for k in range(P.K_shift)]
    label_test_all = list(label_id) # need a new list here
    for ood_class in P.ood_dataset:
        for k in range(P.K_shift):
            feats_all_mean[k].append(feats_ood_np[ood_class][k])
        label_test_all += label_ood[ood_class]

    for k in range(P.K_shift):
        feats_all_mean[k] = np.concatenate(feats_all_mean[k])

    for k in range(P.K_shift):
        print(str(k) + "-th shift")

        print("fitting t-SNE...")
        tsne = manifold.TSNE(n_components=2, random_state=0).fit_transform(feats_all_mean[k])
        train_tsne, test_tsne = np.split(tsne, [train_count])

        print("plotting t-SNE...")
        tsne_fig = visualize_tsne(x_train=train_tsne, y_train=label_train, x_test=test_tsne, y_test=label_test_all, target_class=P.target_class, label_map=label_map)

        print("saving t-SNE...")
        fig_path = prefix + "_tsne" + "_f_" + feature_layer + "_C_" + str(P.target_class) + "_S_" + str(k) + ".png"
        tsne_fig.savefig(fig_path)

    return 


def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

def visualize_tsne(x_train, y_train, x_test, y_test, target_class, label_map: Dict = None) -> Figure:

    x_all = np.concatenate((x_train, x_test))
    x_min, x_max = x_all.min(0), x_all.max(0)

    x_train_norm = (x_train - x_min) / (x_max - x_min)  #Normalize
    x_test_norm = (x_test - x_min) / (x_max - x_min)  #Normalize

    fig_size = 16
    fig = plt.figure(figsize=(fig_size, fig_size))

    for i in range(x_train_norm.shape[0]):
        if label_map is not None:
            y_str = str(label_map[y_train[i]])
        else:
            y_str = str(y_train[i])

        plt.text(x_train_norm[i, 0], x_train_norm[i, 1], y_str, color='k', 
                fontdict={'weight': 'bold', 'size': 9})

    for i in range(x_test_norm.shape[0]):
        if label_map is not None:
            y_str = str(label_map[y_test[i]])
        else:
            y_str = str(y_test[i])
        
        if y_test[i] == target_class:
            color = 'r'
        else:
            color = 'b'
        
        plt.text(x_test_norm[i, 0], x_test_norm[i, 1], y_str, color=color, 
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    # plt.show()

    return fig