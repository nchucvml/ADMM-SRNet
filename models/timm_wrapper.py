import numpy as np
import torch
import torchvision as tv

import timm 

# =======BiT=========
# follow the hyper setting of https://github.com/google-research/big_transfer/tree/master/bit_pytorch

def get_bit_resolution(original_resolution):
  """Takes (H,W) and returns (precrop, crop)."""
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)

bit_known_dataset_sizes = {
  'cifar10': (32, 32),
  'cifar100': (32, 32),
  'svhn': (32, 32),
  'lsun_resize': (32, 32),
  'lsun_fix': (32, 32),
  'imagenet_resize': (32, 32),
  'imagenet_fix': (32, 32),
  
  'oxford_iiit_pet': (224, 224),
  'oxford_flowers102': (224, 224),
  'imagenet2012': (224, 224),
  'imagenet': (224, 224),
  'stanford_dogs': (224, 224),
  'cub': (224, 224),
  'flowers102': (224, 224),
  'places365': (224, 224),
  'food_101': (224, 224),
  'caltech_256': (224, 224),
  'dtd': (224, 224),
  'pets': (224, 224),
}

def get_bit_resolution_from_dataset(dataset):
  if dataset not in bit_known_dataset_sizes:
    raise ValueError(f"Unsupported dataset {dataset}. Add your own here :)")
  return get_bit_resolution(bit_known_dataset_sizes[dataset])

def get_bit_transform(precrop, crop):
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.RandomCrop((crop, crop)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
  
    return train_tx, val_tx

def get_bit_model(num_classes, pretrained=True):
    model = timm.create_model(model_name="resnetv2_50x3_bitm_in21k", pretrained=pretrained, num_classes=num_classes)
    
    return model

def get_bit_optim(model: torch.nn.Module):
    # Note: no weight-decay!
    return torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

def get_bit_mixup(dataset_size):
  return 0.0 if dataset_size < 20_000 else 0.1

def get_bit_schedule(dataset_size):
  if dataset_size < 20_000:
    return [100, 200, 300, 400, 500]
  elif dataset_size < 500_000:
    return [500, 3000, 6000, 9000, 10_000]
  else:
    return [500, 6000, 12_000, 18_000, 20_000]


def get_bit_lr(step, dataset_size, base_lr=0.003):
  """Returns learning-rate for `step` or None at the end."""
  supports = get_bit_schedule(dataset_size)
  # Linear warmup
  if step < supports[0]:
    return base_lr * step / supports[0]
  # End of training
  elif step >= supports[-1]:
    return None
  # Staircase decays by factor of 10
  else:
    for s in supports[1:]:
      if s < step:
        base_lr /= 10
    return base_lr



# =======Swin v2=========