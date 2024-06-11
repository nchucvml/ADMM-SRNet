from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args

from models.cocsr import ContrastiveOneClassSrNet, ContrastiveOneClassSrNetOption
from models.admm_sr import AdmmSrNet

import models.classifier as C

from datasets import set_data_path, get_dataset, get_superclass_list, get_subclass_dataset

P = parse_args()

### Set data path ###
set_data_path(P, P.data_path)

### Set torch device ###

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Initialize dataset ###
ood_eval = (P.mode in ['ood_pre', 
                        'ood_pre_refmix', 'ood_pre_flipmix', 
                        'ood_pre_cocsr', 'ood_pre_cocsr_refmix', 
                        'ood_pre_multi_sr', 'ood_pre_multi_sr_refmix',
                        'ood_pre_sr', 
                        'ood_pre_tsne', 'ood_pre_tsne_shi', 'ood_pre_tsne_dict', 'ood_pre_tsne_dict_refmix', 'ood_pre_tsne_sr',])                          
if (P.dataset == 'imagenet') and ood_eval:
    P.batch_size = 1
    P.test_batch_size = 1
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, eval=ood_eval, download=P.download_data)

P.image_size = image_size
P.n_classes = n_classes

if P.one_class_idx is not None:
    cls_list = get_superclass_list(P.dataset)
    P.n_superclasses = len(cls_list)

    full_test_set = deepcopy(test_set)  # test set of full classes
    train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx])
    test_set = get_subclass_dataset(test_set, classes=cls_list[P.one_class_idx])

if P.dataset != 'imagenet':
    kwargs = {'pin_memory': False, 'num_workers': 4}
else:
    kwargs = {'pin_memory': False, 'num_workers': 0}

train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

if P.ood_dataset is None:
    if P.one_class_idx is not None:
        P.ood_dataset = list(range(P.n_superclasses))
        P.ood_dataset.pop(P.one_class_idx)
    elif P.dataset == 'cifar10':
        P.ood_dataset = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100', 'interp']
    elif P.dataset == 'imagenet':
        P.ood_dataset = ['cub', 'stanford_dogs', 'flowers102', 'places365', 'food_101', 'caltech_256', 'dtd', 'pets']

ood_test_loader = dict()
for ood in P.ood_dataset:
    if ood == 'interp':
        ood_test_loader[ood] = None  # dummy loader
        continue

    if P.one_class_idx is not None:
        ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
        ood = f'one_class_{ood}'  # change save name
    else:
        ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, eval=ood_eval, download=P.download_data)

    ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

### Initialize model ###

simclr_aug, simclr_aug_no_crop = C.get_simclr_augmentation(P, image_size=P.image_size)
simclr_aug = simclr_aug.to(device)
simclr_aug_no_crop = simclr_aug_no_crop.to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
model = C.get_shift_classifer(model, P.K_shift).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if P.mode in ['ood_pre_cocsr', 'ood_pre_cocsr_refmix', 'ood_pre_tsne_dict', 'ood_pre_tsne_dict_refmix']:
    if P.sr_layer == "penultimate":
        feature_dim = model.linear.in_features
        use_z_for_sr = False
    else:
        feature_dim = P.simclr_dim
        use_z_for_sr = True

    dict_num = P.K_shift
    sr_net = AdmmSrNet(dict_num, feature_dim, P.sr_dict_size, P.sr_admm_stage, 
                    initial_rho=P.sr_rho, initial_lambda=P.sr_lambda).to(device)

    cocsr_net_opt = ContrastiveOneClassSrNetOption(use_z_for_sr, P.invert_sr_feature_norm, P.normalize_sr_feature)
    P.cocsr_net = ContrastiveOneClassSrNet(model, sr_net, cocsr_net_opt)

    model = P.cocsr_net

if P.load_path is not None:
    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint, strict=not P.no_strict)
