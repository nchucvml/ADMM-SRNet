# ADMM-SRNet

## ADMM-SRNet: Alternating Direction Method of Multipliers based Sparse Representation Network for One-Class Classification

![image](Overview_ADMM-SRNet.png)

The paper is accepted by [IEEE Transactions on Image Processing, 2023.](https://ieeexplore.ieee.org/document/10124145)

## Usage
The code may contain some redundant parts and will be cleaned up soon.
Performance might vary a little due to randomness.

### Requirements
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+
- tensorboard 2.0+
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 
- [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr) packages 
- [apex](https://github.com/NVIDIA/apex) == 0.1
- [diffdist](https://github.com/ag14774/diffdist) == 0.1 

### Datasets 
Please refer to [CSI (NeurIPS 2020)](https://github.com/alinlab/CSI)

### Training
The training is currently divided in to two stages.
The first stage learns the heterogeneous contrastive feature (HCF) network.
The second stage learns the sparse dictionary (SD) network.

In the following train/eval scripts, you may need to change some arguments to the acutal filepath.
```
--data_path $DATASET_PATH
--resume_path $STAGE1_MODEL_PATH
--load_path $STAGE2_MODEL_PATH
```

For Imagenet and/or multi-class setting, you may need to use multiple GPUs for training/evaluation due to memory limit.
The total batch_size should be 128 (= $GPU_NUM * $BATCH_SIZE).
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --batch_size 32 ...
```

#### CIFAR-10
```
# CIFAR10_Stage1.sh

START=0
END=9
for i in $(seq $START $END);

do 
    python train.py \
    --dataset cifar10 \
    --model resnet18 \
    --mode simclr_CSI \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --batch_size 128 \
    --one_class_idx $i \
    --simclr_dim 512 \
    --suffix f512_mix82
done
```

```
# CIFAR10_Stage2.sh

START=0
END=9
for i in $(seq $START $END);

do 
    python train_cocsr.py \
    --dataset cifar10 \
    --one_class_idx $i \
    --batch_size 128 \
    --model resnet18 \
    --mode cocsr_dict \
    --simclr_dim 512 \
    --sr_lr 1e-3 \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --resume_path ../Data/Model/cifar10_resnet18_unsup_simclr_CSI_shift_rot_bmix_one_class_${i}_f512_mix82 \
    --epoch 1300 \
    --sr_dict_size 512 \
    --sr_lambda 1e-3 \
    --loss_lambda_sr_l1 1e-3 \
    --invert_sr_feature_norm \
    --loss_dict_constr_mode 1 \
    --ini_sr_dict_train \
    --suffix f512_mix82_dm1_init 
done
```

#### CIFAR-10 Multi-class
```
# CIFAR-10 Multi-class Stage 1

python train.py \
--dataset cifar10 \
--model resnet18 \
--mode simclr_CSI \
--shift_trans_type rot_bmix \
--mix_alpha 0.2 \
--batch_size 128 \
--simclr_dim 512 \
--suffix f512_mix82
```

```
# CIFAR-10 Multi-dataset Stage 2

python train_cocsr.py \
--dataset cifar10 \
--batch_size 128 \
--model resnet18 \
--mode cocsr_dict \
--simclr_dim 512 \
--sr_lr 1e-3 \
--shift_trans_type rot_bmix \
--mix_alpha 0.2 \
--resume_path ../Data/Model/cifar10_resnet18_unsup_simclr_CSI_shift_rot_bmix_f512_mix82 \
--epoch 1300 \
--sr_dict_size 512 \
--sr_lambda 1e-3 \
--loss_lambda_sr_l1 1e-3 \
--invert_sr_feature_norm \
--loss_dict_constr_mode 1 \
--ini_sr_dict_train \
--suffix f512_mix82_dm1_init 
```

#### CIFAR-100
```
# CIFAR100_Stage1.sh

START=0
END=19
for i in $(seq $START $END);

do 
    python train.py \
    --dataset cifar100 \
    --model resnet18 \
    --mode simclr_CSI \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --batch_size 128 \
    --one_class_idx $i \
    --simclr_dim 512 \
    --suffix f512_mix82
done
```

```
# CIFAR100_Stage2.sh

START=0
END=19
for i in $(seq $START $END);

do 
    python train_cocsr.py \
    --dataset cifar100 \
    --one_class_idx $i \
    --batch_size 128 \
    --model resnet18 \
    --mode cocsr_dict \
    --simclr_dim 512 \
    --sr_lr 1e-3 \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --resume_path ../Data/Model/cifar100_resnet18_unsup_simclr_CSI_shift_rot_bmix_one_class_${i}_f512_mix82 \
    --epoch 1300 \
    --sr_dict_size 512 \
    --sr_lambda 1e-3 \
    --loss_lambda_sr_l1 1e-3 \
    --invert_sr_feature_norm \
    --loss_dict_constr_mode 1 \
    --ini_sr_dict_train \
    --suffix f512_mix82_dm1_init 
done
```

#### Imagenet
```
# Imagenet_Stage1.sh

START=0
END=29
for i in $(seq $START $END);

do     
    python train.py \
    --dataset imagenet \
    --model resnet18_imagenet \
    --mode simclr_CSI \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --batch_size 128 \
    --one_class_idx $i \
    --simclr_dim 512 \
    --color_jitter_strength 2 \
    --suffix f512_j2_mix82
done
```

```
# Imagenet_Stage2.sh

START=0
END=29
for i in $(seq $START $END);

do 
    python train_cocsr.py \
    --dataset imagenet \
    --one_class_idx $i \
    --batch_size 128 \
    --model resnet18_imagenet \
    --mode cocsr_dict \
    --simclr_dim 512 \
    --sr_lr 1e-3 \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --color_jitter_strength 2 \
    --resume_path ../Data/Model/imagenet_resnet18_imagenet_unsup_simclr_CSI_shift_rot_bmix_one_class_${i}_f512_j2_mix82 \
    --epoch 1300 \
    --sr_dict_size 512 \
    --sr_lambda 1e-3 \
    --loss_lambda_sr_l1 1e-3 \
    --invert_sr_feature_norm \
    --loss_dict_constr_mode 1 \
    --ini_sr_dict_train \
    --suffix f512_j2_mix82_dm1_init  
done
```

#### Imagenet Multi-class
```
# Imagenet Multi-class Stage1

python train.py \
--dataset imagenet \
--model resnet18_imagenet \
--mode simclr_CSI \
--shift_trans_type rot_bmix \
--mix_alpha 0.2 \
--batch_size 128 \
--simclr_dim 512 \
--color_jitter_strength 2 \
--suffix f512_j2_mix82
```

```
# Imagenet Multi-class Stage2

python train_cocsr.py \
--dataset imagenet \
--batch_size 128 \
--model resnet18_imagenet \
--mode cocsr_dict \
--simclr_dim 512 \
--sr_lr 1e-3 \
--shift_trans_type rot_bmix \
--mix_alpha 0.2 \
--color_jitter_strength 2 \
--resume_path ../Data/Model/imagenet_resnet18_imagenet_unsup_simclr_CSI_shift_rot_bmix_f512_j2_mix82 \
--epoch 1300 \
--sr_dict_size 512 \
--sr_lambda 1e-3 \
--loss_lambda_sr_l1 1e-3 \
--invert_sr_feature_norm \
--loss_dict_constr_mode 1 \
--ini_sr_dict_train \
--suffix f512_j2_mix82_dm1_init  
```

### Testing
#### CIFAR-10
```
# CIFAR10_Eval.sh

START=0
END=9
for i in $(seq $START $END);

do 
    python eval.py \
    --dataset cifar10 \
    --one_class_idx $i \
    --model resnet18 \
    --mode ood_pre_cocsr_refmix \
    --simclr_dim 512 \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --load_path ../Data/Model/cifar10_resnet18_unsup_cocsr_dict_shift_rot_bmix_cocsr_ld0.001_lr0.001_invf_one_class_${i}_f512_mix82_dm1_init/last.cocsr_model \
    --ood_score CSI_sr \
    --ood_samples 10 \
    --resize_factor 0.54 \
    --resize_fix \
    --sr_lambda 1e-3 \
    --sr_dict_size 512 \
    --invert_sr_feature_norm \
    --print_score \
    --save_score
done
```

#### CIFAR-10 Multi-class
```
python eval.py \
--dataset cifar10 \
--model resnet18 \
--mode ood_pre_cocsr_refmix \
--simclr_dim 512 \
--shift_trans_type rot_bmix \
--mix_alpha 0.2 \
--load_path ../Data/Model/cifar10_resnet18_unsup_cocsr_dict_shift_rot_bmix_cocsr_ld0.001_lr0.001_invf_f512_mix82_dm1_init/last.cocsr_model \
--ood_score CSI_sr \
--ood_samples 10 \
--resize_factor 0.54 \
--resize_fix \
--sr_lambda 1e-3 \
--sr_dict_size 512 \
--invert_sr_feature_norm \
--print_score \
--save_score
```

#### CIFAR-100
```
# CIFAR100_Eval.sh

START=0
END=19
for i in $(seq $START $END);

do 
    python eval.py \
    --dataset cifar100 \
    --one_class_idx $i \
    --model resnet18 \
    --mode ood_pre_cocsr_refmix \
    --simclr_dim 512 \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --load_path ../Data/Model/cifar100_resnet18_unsup_cocsr_dict_shift_rot_bmix_cocsr_ld0.001_lr0.001_invf_one_class_${i}_f512_mix82_dm1_init/last.cocsr_model \
    --ood_score CSI_sr \
    --ood_samples 10 \
    --resize_factor 0.54 \
    --resize_fix \
    --sr_lambda 1e-3 \
    --sr_dict_size 512 \
    --invert_sr_feature_norm \
    --print_score \
    --save_score
done
```

#### Imagenet
```
# Imagenet_Eval.sh

START=0
END=29
for i in $(seq $START $END);

do 
    python eval.py \
    --dataset imagenet \
    --one_class_idx $i \
    --model resnet18_imagenet \
    --mode ood_pre_cocsr_refmix \
    --simclr_dim 512 \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --load_path ../Data/Model/imagenet_resnet18_imagenet_unsup_cocsr_dict_shift_rot_bmix_cocsr_ld0.001_lr0.001_invf_one_class_${i}_f512_j2_mix82_dm1_init/last.cocsr_model \
    --ood_score CSI_sr \
    --ood_samples 10 \
    --resize_factor 0.54 \
    --resize_fix \
    --sr_lambda 1e-3 \
    --sr_dict_size 512 \
    --invert_sr_feature_norm \
    --print_score \
    --save_score
done
```

#### Imagenet Multi-class
```
python eval.py \
--dataset imagenet \
--model resnet18_imagenet \
--mode ood_pre_cocsr_refmix \
--simclr_dim 512 \
--shift_trans_type rot_bmix \
--mix_alpha 0.2 \
--load_path ../Data/Model/imagenet_resnet18_imagenet_unsup_cocsr_dict_shift_rot_bmix_cocsr_ld0.001_lr0.001_invf_f512_j2_mix82_dm1_init/last.cocsr_model \
--ood_score CSI_sr \
--ood_samples 10 \
--resize_factor 0.54 \
--resize_fix \
--sr_lambda 1e-3 \
--sr_dict_size 512 \
--invert_sr_feature_norm \
--print_score \
--save_score
```

### Visualizatoin
#### CIFAR-10
```
# CIFAR10_TSNE.sh

START=0
END=9
for i in $(seq $START $END);

do 
    python tsne.py \
    --dataset cifar10 \
    --one_class_idx $i \
    --model resnet18 \
    --mode ood_pre_tsne_dict_refmix \
    --simclr_dim 512 \
    --shift_trans_type rot_bmix \
    --mix_alpha 0.2 \
    --load_path ../Data/Model/cifar10_resnet18_unsup_cocsr_dict_shift_rot_bmix_cocsr_ld0.001_lr0.001_invf_one_class_${i}_f512_mix82_dm1_init/last.cocsr_model \
    --ood_layer penultimate \
    --ood_score CSI_sr \
    --ood_samples 10 \
    --resize_factor 0.54 \
    --resize_fix \
    --sr_lambda 1e-3 \
    --sr_dict_size 512 \
    --invert_sr_feature_norm 
done
```

## Comparison with State-of-the-art Methods (%)

| Method            | Dataset           |  AUROC (Mean) |
| ------------------|------------------ | --------------|
| SimCLR            | CIFAR-10-OC       |      87.9%    |
| Rot+Trans         | CIFAR-10-OC       |      89.8%    |
| CSI               | CIFAR-10-OC       |      94.3%    |
| ADMM-SRNet (ours) | CIFAR-10-OC       |      95.4%    |

| Method            | Dataset           |  AUROC (Mean) |
| ------------------|------------------ | --------------|
| SimCLR            | CIFAR-10-OC       |      87.9%    |
| Rot+Trans         | CIFAR-10-OC       |      89.8%    |
| CSI               | CIFAR-10-OC       |      94.3%    |
| ADMM-SRNet (ours) | CIFAR-10-OC       |      95.4%    |

For more detail, please refer to our paper.

## Trained Models
[Google Drive](https://drive.google.com/drive/folders/1z0eTFlGp6aekYhOcN_oRZTtUD8wdjcmh?usp=sharing)

## Citation 
Please cite the following paper when you apply the code. 

[1] C.-Y. Chiou, K.-T. Lee, C.-R. Huang and P.-C. Chung, "ADMM-SRNet: Alternating Direction Method of Multipliers based Sparse Representation Network for One-Class Classification," IEEE Transactions on Image Processing, vol. 32, pp. 2843-2856, 2023, doi: 10.1109/TIP.2023.3274488.

[2] K.-T. Lee, C.-Y. Chiou and C.-R. Huang, "One-Class Novelty Detection via Sparse Representation with Contrastive Deep Features," in Proc. International Computer Symposium (ICS), 2020, pp. 61-66, doi: 10.1109/ICS51289.2020.00022.

## Acknowledgement
This work was supported in part by the National Science and Technology Council of Taiwan under Grant NSTC 111-2634-F-006-012, NSTC 111-2628-E-006-011-MY3, NSTC 112-2622-8-006-009-TE1, and MOST 111-2327-B-006-007.

We would like to thank the National Center for High-Performance Computing (NCHC) Taiwan for providing computational and storage resources.

Part of the code is modified from the official release of 
 "CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances" (NeurIPS 2020) by Jihoon Tack*, Sangwoo Mo*, Jongheon Jeong, and Jinwoo Shin.
https://github.com/alinlab/CSI
