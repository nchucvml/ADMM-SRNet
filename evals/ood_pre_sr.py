import os
from copy import deepcopy

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sporco.dictlrn import bpdndl
from sporco.admm import bpdn

from tqdm import tqdm

import models.transform_layers as TL
from utils.utils import set_random_seed, normalize
from evals.evals import get_auroc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
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

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': [feature_layer], #P.ood_layer,
    }

    print('Pre-compute global statistics...')
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)

    P.axis = []
    for f in feats_train[feature_layer].chunk(P.K_shift, dim=1):
        axis = f.mean(dim=1)  # (M, d)
        P.axis.append(normalize(axis, dim=1).to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))
    
    # P.sr_lambda = 0.01
    # P.sr_dict_size = 512
    time_start = time.perf_counter()
    P.dict = get_dictionary(P.axis[0], dict_size=P.sr_dict_size, sr_lambda=P.sr_lambda)
    time_end = time.perf_counter()
    print('dictionary build time {}'.format((time_end - time_start)))

    print('Pre-compute features...')
    feats_id = get_features(P, P.dataset, model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    feats_ood = dict()
    for ood, ood_loader in ood_loaders.items():
        if ood == 'interp':
            feats_ood[ood] = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, **kwargs)
        else:
            feats_ood[ood] = get_features(P, ood, model, ood_loader, prefix=prefix, **kwargs)

    print(f'Compute OOD scores... (score: {ood_score})')
    # TODO: save sr results 
    time_start = time.perf_counter()
    scores_id = get_scores(P, feats_id, ood_score, feature_layer).numpy()
    scores_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []

    for ood, feats in feats_ood.items():
        scores_ood[ood] = get_scores(P, feats, ood_score, feature_layer).numpy()
        auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood[ood])
        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])
    
    time_end = time.perf_counter()
    print('one class classification time {}'.format((time_end - time_start)))

    if P.one_class_idx is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    return auroc_dict


def get_scores(P, feats_dict, ood_score, feature_layer="simclr"):
    # convert to gpu tensor
    feats_sim = feats_dict[feature_layer].to(device)
    # feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    print(feats_sim[0].shape)

    # compute scores
    scores = []
    for f_sim in tqdm(feats_sim):
        f_sim = [normalize(f.mean(dim=0), dim=0) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)        
        score = 0
        
        S0 = np.expand_dims(f_sim[0].cpu().numpy().T, axis=1)

        # params setting
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                                'RelStopTol': 1e-3, 'AutoRho': {'RsdlTarget': 1.0}})
        
        # Compute sparse representation
        bp = bpdn.BPDN(P.dict, S0, P.sr_lambda, opt)
        bp.solve()
        score = 1 - bp.itstat[-1].DFid

        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


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

def get_dictionary(features, dict_size, sr_lambda):
    print("Dictionary lerning...")

    features = features.cpu().numpy().T
    print(features.shape)

    # Init dictionary
    #dict_size = 128 # for experiment, use differnet size to the trained network
    D0 = np.random.randn(features.shape[0], dict_size)

    # Params setting
    lmbda = sr_lambda
    opt = bpdndl.BPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 300,
                                        'BPDN': {'rho': 10.0 * lmbda + 0.1},
                                        'CMOD': {'rho': features.shape[1] / 1e3}})
    d = bpdndl.BPDNDictLearn(D0, features, lmbda, opt)

    # Compute final dictionary
    d.solve()
    print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))
    dictionary = d.getdict()
    print("dict shape " + str(dictionary.shape))
    return dictionary


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

