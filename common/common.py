from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['cifar10', 'cifar100', 'imagenet'], type=str)
    parser.add_argument('--data_path', help='Path to data folder',
                        default='../Data/', type=str)    
    parser.add_argument("--download_data", help='download data',
                        action='store_true', default=False)
    parser.add_argument('--num_worker', help='Number of worker for training',
                        default=4, type=int)

    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--model', help='Model',
                        choices=['resnet18', 'resnet18_imagenet', 'timm_resnetv2_bit', 'timm_swinv2'], type=str)
    parser.add_argument("--pretrained_model", help='use pretrained model weight, only for bit and swin model',
                        action='store_true', default=False)
    parser.add_argument('--mode', help='Training mode',
                        default='simclr', type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',
                        choices=['rotation', 'cutperm', 'bmix',
                                 'rot_bmix', 'rot_smix', 'rot_perm', 
                                 'rot_cropmix', 'rot_augmix',
                                 'rot_perm_bmix',
                                 'none'], type=str)
    parser.add_argument('--mix_alpha', help='for mix-up augmentation, Image = img1 * (1 - alpha) + img2 * alpha',
                        default=0.5, type=float)
    parser.add_argument('--color_jitter_strength', help='for color jitter augmentation, range = strength * 0.4',
                        default=1, type=float)     
    parser.add_argument("--cropmix_resize_factor", help='resize scale for cropmix is sampled from [cropmix_resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--cropmix_resize_fix", help='cropmix resize scale is fixed to cropmix_resize_factor (not (cropmix_resize_factor, 1.0])',
                        action='store_true')    
    parser.add_argument("--cropmix_crop_aug", help='crop again when generating simclr aug for cropmix shifting',
                        action='store_true')            

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=5, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=10, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'adam', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)

    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['norm_mean'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=1, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')
    parser.add_argument("--refmix_batch_size", help='reference batch size for refmix eval',
                        default=128, type=int)

    # Sparse reconstruction
    parser.add_argument('--sr_lambda', help='L1 constraint weight for sparse reconstruction',
                        default=0.01, type=float)
    parser.add_argument("--sr_dict_size", help='dictionary size of sparse reconstruction',
                        default=512, type=int)  
    parser.add_argument("--sr_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr'],
                        default='penultimate', type=str)     
    parser.add_argument("--invert_sr_feature", help='f -> f.max - f',
                        action='store_true')       
    parser.add_argument("--invert_sr_feature_norm", help='f -> f / |f|_2^4',
                        action='store_true')        
    parser.add_argument("--invert_sr_feature_mean", help='f -> f.mean - f',
                        action='store_true')    
    parser.add_argument("--invert_sr_feature_DNT", help='f -> DNT(f)',
                        action='store_true')     
    parser.add_argument("--invert_sr_coef_DNT", help='a -> DNT(a)',
                        action='store_true')  
    parser.add_argument("--ini_sr_dict_train", help='initialize sr dict using features of training data',
                        action='store_true')                     

    parser.add_argument("--normalize_sr_feature", help='normalize features for sr',
                        action='store_true')      
    parser.add_argument("--normalize_sr_dict_feature", help='normalize features for sr dictionary learning',
                        action='store_true')        

    # ADMM SR
    parser.add_argument('--sr_rho', help='parameter rho in ADMM sr',
                        default=1, type=float)
    parser.add_argument('--sr_admm_stage', default=40, type=int,
                        help='Number of admm stage for solving lasso sparse recontruction (default 40)') 
    parser.add_argument('--sr_lr', type=float, default=1e-3,
                        help='Learning rate for sr network (default: 1e-3)')    
    parser.add_argument("--sr_decay_lr", default=1e-6, action="store", type=float,
                        help='Learning rate decay (default: 1e-6)') 

    parser.add_argument('--loss_lambda_sr', type=float, default=1.0,
                              help='Weight of sparse reconstruction loss (default: 1.0)')
    parser.add_argument('--loss_lambda_sr_l1', type=float, default=0.01,
                            help='Weight of sparse reconstruction coefficient loss (default: 0.01)')
    parser.add_argument('--loss_lambda_dict_constr', type=float, default=0.1,
                            help='Weight of dictionary constraint loss (default: 0.1)')
    parser.add_argument('--loss_dict_constr_mode', type=int, default=0,
                            help='Dictionary constraint loss mode. mode 0: |d_m|_2 = 1. mode 1: |d_m|_2 <= 1. (default: 0)')    
    parser.add_argument('--loss_dict_constr_value', type=float, default=1.0,
                            help='Constraint value of dictionary atom (default: 1.0)')                        
    parser.add_argument('--loss_lambda_dict_contra_sr', type=float, default=1,
                            help='Weight of dictionary contrastive sr loss (default: 1)')
    
    parser.add_argument('--loss_sr_deep_supervision', help='delimited list input of sr deep supervision stages', 
                            type=lambda s: [int(item) for item in s.split(',')], default=[])
                                
    parser.add_argument('--loss_dict_detach_feature', action='store_true', default=False,
                            help='Detach features when learning dict, only in experiment program (default: False)')  
    parser.add_argument('--loss_dict_detach_sr_coef', action='store_true', default=False,
                            help='Detach SR coef when learning dict (default: False)') 
                            
    # scores
    parser.add_argument("--disable_sim_score", help='set sim score to 0',
                        action='store_true')   
    parser.add_argument("--disable_shi_score", help='set shi score to 0',
                        action='store_true')                                                                     

    parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
