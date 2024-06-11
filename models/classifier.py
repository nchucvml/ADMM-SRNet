import numpy as np

import torch.nn as nn

from models.resnet import ResNet18, ResNet34, ResNet50
from models.resnet_imagenet import resnet18, resnet50
import models.transform_layers as TL


def get_simclr_augmentation(P, image_size):

    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor
    if P.resize_fix: # if resize_fix is True, use same scale
        resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4 * P.color_jitter_strength, 
                                       contrast=0.4 * P.color_jitter_strength, 
                                       saturation=0.4 * P.color_jitter_strength, 
                                       hue=0.1 * P.color_jitter_strength, 
                                       p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    if P.dataset == 'imagenet': # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    transform_no_crop = nn.Sequential(
                            color_jitter,
                            color_gray,
                        )

    return transform, transform_no_crop


def get_shift_module(P, eval=False):
    if P.shift_trans_type == 'rot_bmix':
        rot = TL.Rotation()
        k_rot = 4

        mix = TL.BatchMix(P.mix_alpha)
        k_mix = 1

        shift_transform = TL.TransformList([rot, mix], [k_rot, k_mix])
        K_shift = np.sum([k_rot, k_mix])
    
    elif P.shift_trans_type == 'rot_smix':
        rot = TL.Rotation()
        k_rot = 4

        mix = TL.ShiftMix(rot, k_rot, P.mix_alpha)
        k_mix = 3

        shift_transform = TL.TransformList([rot, mix], [k_rot, k_mix])
        K_shift = np.sum([k_rot, k_mix])

    elif P.shift_trans_type == 'rot_cropmix':
        rot = TL.Rotation()
        k_rot = 4

        # parameter for resizecrop
        resize_scale = (P.cropmix_resize_factor, 1.0) # resize scaling factor
        if P.cropmix_resize_fix: # if resize_fix is True, use same scale
            resize_scale = (P.cropmix_resize_factor, P.cropmix_resize_factor)
        mix = TL.CropMix(P.mix_alpha, scale=resize_scale, size=P.image_size)
        k_mix = 1

        shift_transform = TL.TransformList([rot, mix], [k_rot, k_mix])
        K_shift = np.sum([k_rot, k_mix])

    elif P.shift_trans_type == 'rot_augmix':
        rot = TL.Rotation()
        k_rot = 4
        
        from types import SimpleNamespace
        param = SimpleNamespace()
        param.resize_factor = P.cropmix_resize_factor
        param.resize_fix = P.cropmix_resize_fix
        param.color_jitter_strength = P.color_jitter_strength
        param.dataset = P.dataset

        aug, _ = get_simclr_augmentation(param, image_size=P.image_size)

        mix = TL.AugMix(aug, P.mix_alpha)
        k_mix = 1

        shift_transform = TL.TransformList([rot, mix], [k_rot, k_mix])
        K_shift = np.sum([k_rot, k_mix])

    elif P.shift_trans_type == 'rot_perm':
        rot = TL.Rotation()
        k_rot = 4

        perm = TL.CutPerm()
        k_perm = 3

        shift_transform = TL.TransformList([rot, perm], [k_rot, k_perm], [0, 1])
        K_shift = np.sum([k_rot, k_perm])
    
    elif P.shift_trans_type == 'rot_perm_bmix':
        rot = TL.Rotation()
        k_rot = 4

        perm = TL.CutPerm()
        k_perm = 3
        
        mix = TL.BatchMix(P.mix_alpha)
        k_mix = 1

        shift_transform = TL.TransformList([rot, perm, mix], [k_rot, k_perm, k_mix], [0, 1, 0])
        K_shift = np.sum([k_rot, k_perm, k_mix])

    elif P.shift_trans_type == 'rotation':
        shift_transform = TL.Rotation()
        K_shift = 4

    elif P.shift_trans_type == 'cutperm':
        shift_transform = TL.CutPerm()
        K_shift = 4

    elif P.shift_trans_type == 'bmix':
        id = TL.Identity()
        k_id = 1

        mix = TL.BatchMix(P.mix_alpha)
        k_mix = 1

        shift_transform = TL.TransformList([id, mix], [k_id, k_mix])
        K_shift = np.sum([k_id, k_mix])

    else:
        shift_transform = nn.Identity()
        K_shift = 1

    if not eval and not ('sup' in P.mode):
        assert P.batch_size == int(128/K_shift)

    return shift_transform, K_shift


def get_shift_classifer(model, K_shift):

    model.shift_cls_layer = nn.Linear(model.last_dim, K_shift)

    return model


def get_classifier(mode, n_classes=10):
    if mode == 'resnet18':
        classifier = ResNet18(num_classes=n_classes)
    elif mode == 'resnet34':
        classifier = ResNet34(num_classes=n_classes)
    elif mode == 'resnet50':
        classifier = ResNet50(num_classes=n_classes)
    elif mode == 'resnet18_imagenet':
        classifier = resnet18(num_classes=n_classes)
    elif mode == 'resnet50_imagenet':
        classifier = resnet50(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier

