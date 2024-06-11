from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from models import timm_wrapper
from models.cocsr import ContrastiveOneClassSrNet, ContrastiveOneClassSrNetOption
from models.admm_sr import AdmmSrNet

from training.cocsr import CocsrDictionaryLearningLoss, CocsrDictionaryLearningLossOption
from training.sr import SparseDictionaryConstraintLossOption

from common.common import parse_args
import models.classifier as C
from datasets import set_data_path, get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import load_checkpoint

P = parse_args()

### Set data path ###
set_data_path(P, P.data_path)

### Set torch device ###

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()

if P.n_gpus > 1:
    import apex
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    P.multi_gpu = True
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=P.n_gpus,
        rank=P.local_rank,
    )
else:
    P.multi_gpu = False

### only use one ood_layer while training
P.ood_layer = P.ood_layer[0]

### Initialize dataset ###
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, download=P.download_data)
P.image_size = image_size

# these two is using supervised training with only a single class for unsupervised mode
if P.mode in ['bit', 'swin']:   
    P.n_classes = 1
else:
    P.n_classes = n_classes

if P.one_class_idx is not None:
    cls_list = get_superclass_list(P.dataset)
    P.n_superclasses = len(cls_list)

    full_test_set = deepcopy(test_set)  # test set of full classes
    train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx])
    test_set = get_subclass_dataset(test_set, classes=cls_list[P.one_class_idx])

if P.dataset != 'imagenet':
    kwargs = {'pin_memory': False, 'num_workers': P.num_worker}
else:
    kwargs = {'pin_memory': False, 'num_workers': 0}

if P.multi_gpu:
    train_sampler = DistributedSampler(train_set, num_replicas=P.n_gpus, rank=P.local_rank)
    test_sampler = DistributedSampler(test_set, num_replicas=P.n_gpus, rank=P.local_rank)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=P.test_batch_size, **kwargs)
else:
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

if P.ood_dataset is None:
    if P.one_class_idx is not None:
        P.ood_dataset = list(range(P.n_superclasses))
        P.ood_dataset.pop(P.one_class_idx)
    elif P.dataset == 'cifar10':
        P.ood_dataset = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100', 'interp']
    elif P.dataset == 'imagenet':
        P.ood_dataset = ['cub', 'stanford_dogs', 'flowers102']

ood_test_loader = dict()
for ood in P.ood_dataset:
    if ood == 'interp':
        ood_test_loader[ood] = None  # dummy loader
        continue

    if P.one_class_idx is not None:
        ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
        ood = f'one_class_{ood}'  # change save name
    else:
        ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, download=P.download_data)

    if P.multi_gpu:
        ood_sampler = DistributedSampler(ood_test_set, num_replicas=P.n_gpus, rank=P.local_rank)
        ood_test_loader[ood] = DataLoader(ood_test_set, sampler=ood_sampler, batch_size=P.test_batch_size, **kwargs)
    else:
        ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

### Initialize model ###

simclr_aug, simclr_aug_no_crop  = C.get_simclr_augmentation(P, image_size=P.image_size)
simclr_aug = simclr_aug.to(device)
simclr_aug_no_crop = simclr_aug_no_crop.to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

if P.mode == 'bit':    
    P.bit_mixup = timm_wrapper.get_bit_mixup(len(train_set))
    P.bit_mixup_l = np.random.beta(P.bit_mixup, P.bit_mixup) if P.bit_mixup > 0 else 1

if P.mode == 'bit':   
    model = timm_wrapper.get_bit_model(P.n_classes, P.pretrained_model).to(device)
elif P.mode == 'swin':
    raise NotImplementedError()
else:
    model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
    model = C.get_shift_classifer(model, P.K_shift).to(device)

if P.mode in ['bit', 'swin']:
    criterion = nn.BCEWithLogitsLoss().to(device)
else:
    criterion = nn.CrossEntropyLoss().to(device)

if P.mode == 'bit':   
    optimizer = timm_wrapper.get_bit_optim(model)
elif P.mode == 'swin':
    raise NotImplementedError()
elif P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=P.lr_init, betas=(.9, .999), weight_decay=P.weight_decay)
    lr_decay_gamma = 0.3
elif P.optimizer == 'lars':
    from torchlars import LARS
    base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()

if P.mode == 'bit':   
    P.bit_base_lr = 0.003

if P.mode == 'swin':
    raise NotImplementedError()

if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

from training.scheduler import GradualWarmupScheduler
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

if P.mode == "cocsr_dict":
    # initialize models 
    # TODO: this is not good =(, try to make common functions for this
    if P.sr_layer == "penultimate":
        feature_dim = model.linear.in_features
        use_z_for_sr = False
    else:
        feature_dim = P.simclr_dim
        use_z_for_sr = True

    dict_num = P.K_shift
    sr_net = AdmmSrNet(dict_num, feature_dim, P.sr_dict_size, P.sr_admm_stage, 
                    initial_rho=P.sr_rho, initial_lambda=P.sr_lambda, dict_constraint=P.loss_dict_constr_value).to(device)

    cocsr_net_opt = ContrastiveOneClassSrNetOption(use_z_for_sr, P.invert_sr_feature_norm, P.normalize_sr_dict_feature)
    P.cocsr_net = ContrastiveOneClassSrNet(model, sr_net, cocsr_net_opt)

    # optimizer
    P.dict_optim = torch.optim.Adam(P.cocsr_net.sr_nets.parameters(), lr=P.sr_lr, weight_decay=P.sr_decay_lr)
    # don't use scheduler, keep constant learning rate

    # loss
    cocsr_loss_op = CocsrDictionaryLearningLossOption(                     
                        compute_sr_loss=(P.loss_lambda_sr > 0), 
                        loss_lambda_sr=P.loss_lambda_sr,
                        detach_sr_coef_for_sr=P.loss_dict_detach_sr_coef,

                        compute_sr_l1_loss=(P.loss_lambda_sr_l1 > 0), 
                        loss_lambda_sr_l1=P.loss_lambda_sr_l1,

                        compute_dict_constr_loss=(P.loss_lambda_dict_constr > 0), 
                        loss_lambda_dict_constr=P.loss_lambda_dict_constr,
                        dict_constr_loss_op=SparseDictionaryConstraintLossOption(
                                                constr_mode=P.loss_dict_constr_mode, 
                                                constraint=P.loss_dict_constr_value),
                        
                        compute_sr_deep_supervision=False,
                                                    
                        verbose=True,
                        debug=False,
                        )
    P.dict_loss = CocsrDictionaryLearningLoss.from_option(cocsr_loss_op)

if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=not P.no_strict)
    
    if P.mode != 'cocsr_dict':
        optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    
    if 'best' in config:
        best = config['best']
    else:
        best = 100.0
    error = 100.0
else:
    resume = False
    start_epoch = 1
    best = 100.0
    error = 100.0

if P.mode == 'sup_linear' or P.mode == 'sup_CSI_linear':
    assert P.load_path is not None
    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint, strict=not P.no_strict)

if P.multi_gpu:
    simclr_aug = apex.parallel.DistributedDataParallel(simclr_aug, delay_allreduce=True)
    model = apex.parallel.convert_syncbn_model(model)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
