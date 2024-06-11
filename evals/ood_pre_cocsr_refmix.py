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
from evals.sr import compute_reconstruction_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None, simclr_aug_no_crop=None):
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

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': ['sr_feature', 'sr_coef', 'shift'],
    }

    print('Pre-compute global statistics...')
    ref_count = 0
    ref_batch = []
    while ref_count < P.refmix_batch_size:
        if P.dataset == 'imagenet':
            # image net process one image at once, the batch are augmentations of the same image 
            tmp = (next(iter(train_loader)))[0][1].to(device)
            ref_batch.append(tmp)
            ref_count += tmp.shape[0] 
        else:
            tmp = (next(iter(train_loader)))[0].to(device)
            ref_batch.append(tmp)
            ref_count += tmp.shape[0]
    ref_batch = torch.cat(ref_batch, dim=0)

    time_start = time.perf_counter()
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)
    time_end = time.perf_counter()
    print('train feature time {}'.format((time_end - time_start)))

    #dictionary
    time_start = time.perf_counter()

    all_D = model.sr_nets.get_dictionary()
    P.dict = [D for D in all_D]
    
    P.axis = []
    for f in feats_train['sr_feature'].chunk(P.K_shift, dim=1):
        axis = f.mean(dim=1).to(device)  # (M, d)        
        P.axis.append(axis)

    time_end = time.perf_counter()
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))
    print('dictionary build time {}'.format((time_end - time_start)))

    # compute training set sr
    time_start = time.perf_counter()
    f_train_chunks = feats_train['sr_feature'].chunk(P.K_shift, dim=1)
    all_error = []    
    for shift_idx, D in enumerate(P.dict):
        f = f_train_chunks[shift_idx]
        feature = f.mean(dim=1).to(device)  # (M, d)                
        feature = feature.t()

        sr_coef_chunk = feats_train['sr_coef'][shift_idx].chunk(P.K_shift, dim=1)
        sr_coef = sr_coef_chunk[shift_idx]       
        sr_coef = sr_coef.mean(dim=1).to(device) # (M, d)  
        sr_coef = sr_coef.t()

        sr_err = get_sr_error(feature, P.dict[shift_idx], sr_coef)
        
        all_error.append(torch.Tensor(sr_err))
    time_end = time.perf_counter()
    print('training set sr time {}'.format((time_end - time_start)))

    # mean value of training samples for normalizing
    # f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    weight_sim = []
    weight_shi = []
    w_sim_mask = 0 if P.disable_sim_score else 1
    w_shi_mask = 0 if P.disable_shi_score else 1
    for shi in range(P.K_shift):
        sim_norm = all_error[shi]  # (M)
        shi_mean = f_shi[shi][:, shi]  # (M)

        w_sim = (1 / sim_norm.mean().item()) * w_sim_mask
        w_shi = (1 / shi_mean.mean().item()) * w_shi_mask

        weight_sim.append(w_sim)
        weight_shi.append(w_shi)

    if ood_score == 'simclr_sr':
        P.weight_sim = [1] + [0] * max(0, P.K_shift - 1)
        P.weight_shi = [0] * P.K_shift
    elif ood_score == 'CSI_sr':
        P.weight_sim = weight_sim
        P.weight_shi = weight_shi
    else:
        raise ValueError()

    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))

    print('Compute features and OOD scores... (score: {ood_score})...')
    time_start = time.perf_counter()

    feats_id = get_features(P, P.dataset, model, id_loader, prefix=prefix, ref_batch=ref_batch, **kwargs)  # (N, T, d)
    scores_id = get_scores(P, feats_id, ood_score, 'sr_feature').numpy()
    feats_id = None # delete features to reduce memory usage
    
    scores_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []
    for ood, ood_loader in ood_loaders.items():
        if ood == 'interp':
            feats = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, ref_batch=ref_batch, **kwargs)
        else:
            feats = get_features(P, ood, model, ood_loader, prefix=prefix, ref_batch=ref_batch, **kwargs)
    
        scores_ood[ood] = get_scores(P, feats, ood_score, 'sr_feature').numpy()
        auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood[ood])
        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])

        feats = None # delete features to reduce memory usage
    
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
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    print(feats_sim[0].shape)

    # compute sr
    all_sr_coef = []
    all_feature = []
    all_error = []
    for shift_idx, f in enumerate(feats_sim.chunk(P.K_shift, dim=1)):
        feature = f.mean(dim=1)  # (M, d)        
        feature = feature.t()

        sr_coef_chunk = feats_dict['sr_coef'][shift_idx].chunk(P.K_shift, dim=1)
        sr_coef = sr_coef_chunk[shift_idx]       
        sr_coef = sr_coef.mean(dim=1).to(device) # (M, d)  
        sr_coef = sr_coef.t()
        
        sr_err = get_sr_error(feature, P.dict[shift_idx], sr_coef)
        
        all_sr_coef.append(sr_coef)
        all_feature.append(feature)
        all_error.append(sr_err)

    # compute scores
    scores = []
    for f_idx in tqdm(range(N)):  
        f_shi = [f.mean(dim=0, keepdim=True) for f in feats_shi[f_idx].chunk(P.K_shift)]  # list of (1, 4)     
        score = 0
        
        for shift_idx in range(P.K_shift):            
            err = all_error[shift_idx][0, f_idx]   
            if P.invert_sr_coef_DNT and not P.invert_sr_feature:       
                err *= -1

            score += -(err * P.weight_sim[shift_idx])

            score += f_shi[shift_idx][:, shift_idx].item() * P.weight_shi[shift_idx]

        score = score / P.K_shift
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift'), ref_batch=None):

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
                                    simclr_aug, sample_num, layers=left, ref_batch=ref_batch)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            # torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift'), ref_batch=None):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    if 'sr_coef' in layers:
        feats_all['sr_coef'] = [[] for _ in range(P.K_shift)] 
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
        if 'sr_coef' in layers:
            feats_batch['sr_coef'] = [[] for _ in range(P.K_shift)] 
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1 and 'bmix' in P.shift_trans_type and ref_batch is not None:
                ref_b = ref_batch.size()[0]
                ref_i = torch.randint(0, ref_b, (1,))

                x_temp = [P.shift_trans(hflip(x), k) for k in range(P.K_shift - 1)]
                x_mix = hflip(x) * (1- P.mix_alpha) + hflip(ref_batch[ref_i]) * P.mix_alpha
                x_temp.append(x_mix)
                x_t = torch.cat(x_temp)
            elif P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, transpose_sr_coef=True, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if layer != 'sr_coef':                    
                    if imagenet is False:
                        feats_batch[layer] += feats.chunk(P.K_shift)
                    else:
                        feats_batch[layer] += [feats]  # (B, d) cpu tensor
                else:
                    if imagenet is False:
                        for i in range(P.K_shift):
                            feats_batch[layer][i] += (feats[i].chunk(P.K_shift))
                    else:
                        for i in range(P.K_shift):
                            feats_batch[layer][i] += [feats[i]]

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if key != 'sr_coef': 
                if imagenet:
                    feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
                else:
                    feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)
            else:
                for i in range(P.K_shift):
                    if imagenet:
                        feats_batch[key][i] = torch.stack(val[i], dim=0)  # (B, T, d)
                    else:
                        feats_batch[key][i] = torch.stack(val[i], dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            if layer != 'sr_coef':
                feats_all[layer] += [feats_batch[layer]]
            else:
                for i in range(P.K_shift):
                    feats_all[layer][i] += [feats_batch[layer][i]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        if key != 'sr_coef':
            feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)
        else:
            for i in range(P.K_shift):
                feats_all[key][i] = torch.cat(val[i], dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            if key != 'sr_coef':
                N, T, d = val.size()  # T = K * T'
                val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
                val = val.transpose(2, 1)  # (N, 4, T', d)
                val = val.reshape(N, T, d)  # (N, T, d)
                feats_all[key] = val
            else:
                for i in range(P.K_shift):
                    vi = val[i]
                    N, T, d = vi.size()  # T = K * T'
                    vi = vi.view(N, -1, P.K_shift, d)  # (N, T', K, d)
                    vi = vi.transpose(2, 1)  # (N, 4, T', d)
                    vi = vi.reshape(N, T, d)  # (N, T, d)
                    feats_all[key][i] = vi

    return feats_all

def get_sr_error(feature, sr_dict, sr_coef):
    """
    Args:
        feature: dim x N
        sr_dict: dim x DN
        sr_coef: DN x N

    Returns:
        err: 1 x N
    """

    f_t = torch.as_tensor(feature)
    sr_t = torch.as_tensor(sr_coef)
    dict_t = torch.as_tensor(sr_dict)

    err_t = compute_reconstruction_error(dict_t, sr_t, f_t, keepdim=True)

    return err_t.cpu()

def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

