import time

import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.cluster import KMeans

import models.transform_layers as TL

from utils.utils import AverageMeter, normalize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None, simclr_aug_no_crop=None):

    assert simclr_aug is not None
    assert simclr_aug_no_crop is not None

    model.eval()

    if P.multi_gpu:
        pass
    else:
        pass

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if epoch == 1000:
        # initilaize dictionary with training features
        # the epoch number should not be hard coded
        if P.ini_sr_dict_train:
            ini_sr_dict(P, model, loader, simclr_aug, log_)       

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['sr'] = AverageMeter()
    losses['sr_l1'] = AverageMeter()
    losses['sr_dict'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):        
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        with torch.no_grad():
            ### SimCLR loss ###
            if P.dataset != 'imagenet':
                batch_size = images.size(0)
                images = images.to(device)
                images = hflip(images)  # 2B with hflip
            else:
                batch_size = images[0].size(0)
                images = images[0].to(device)

            labels = labels.to(device)

            
            if P.shift_trans_type == 'rot_cropmix':
                # crop mix augmentations are already cropped and does not need addictionarl crop

                images_rot = torch.cat([P.shift_trans(images, k) for k in range(P.K_shift-1)])
                images_cmix = P.shift_trans(images, P.K_shift-1)

                # transform
                images_rot = simclr_aug(images_rot)  
                if P.cropmix_crop_aug:
                    images_cmix = simclr_aug(images_cmix)
                else:
                    images_cmix = simclr_aug_no_crop(images_cmix) 

                images = torch.cat([images_rot, images_cmix], dim=0)  # KB
            
            elif P.shift_trans_type == 'rot_augmix':
                # rotation
                images_rot = torch.cat([P.shift_trans(images, k) for k in range(P.K_shift-1)])
                images_rot = simclr_aug(images_rot)  

                # augmix
                images_amix = P.shift_trans(images, P.K_shift-1)

                # concat
                images = torch.cat([images_rot, images_amix], dim=0)  # KB
            
            else: # other cases
                images = torch.cat([P.shift_trans(images, k) for k in range(P.K_shift)])                
                images = simclr_aug(images)  # simclr augmentation

        shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_shift)], 0)  # B -> 4B
                
        _, outputs_aux = P.cocsr_net(images, sr_feature=True, sr_coef=True)
        sr_feature = outputs_aux['sr_feature'].detach()
        sr_coef = outputs_aux['sr_coef']
        all_dict = P.cocsr_net.sr_nets.get_dictionary()

        ### dict loss ###
        loss_dict, loss_record = P.dict_loss(shift_labels, sr_feature, sr_coef, [], all_dict)

        P.dict_optim.zero_grad()
        loss_dict.backward()
        P.dict_optim.step()

        ### optimizer learning rate ###
        lr = P.dict_optim.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['sr'].update(loss_record.sr_reconstruction_loss, batch_size)
        losses['sr_l1'].update(loss_record.sr_l1_loss, batch_size)
        losses['sr_dict'].update(loss_record.sr_dict_constr_loss, batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossSR %f] [LossSRL1 %f] [LossSRDict %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['sr'].value, losses['sr_l1'].value, losses['sr_dict'].value))
        check = time.time()

    # P.linear_scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossSR %f] [LossSRL1 %f] [LossSRDict %f]' %
         (batch_time.average, data_time.average,
          losses['sr'].average, losses['sr_l1'].average, losses['sr_dict'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_sr', losses['sr'].average, epoch)
        logger.scalar_summary('train/loss_sr_l1', losses['sr_l1'].average, epoch)
        logger.scalar_summary('train/loss_sr_dict', losses['sr_dict'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)


def ini_sr_dict(P, model, loader, simclr_aug, log_func):
    # get features 
    log_func("compute features")
    feats_all = [[] for _ in range(P.K_shift)]
    with torch.no_grad():
        for _, (images, _) in enumerate(loader):     

            ### SimCLR ###
            if P.dataset != 'imagenet':
                images = images.to(device)
                images = hflip(images)  # hflip
            else: # imagenet
                images = images[0].to(device)

            images = torch.cat([P.shift_trans(images, k) for k in range(P.K_shift)])
            
            images = simclr_aug(images)  # simclr augmentation
            _, outputs_aux = P.cocsr_net(images, sr_feature=True, sr_coef=False)
            feats = outputs_aux['sr_feature'].cpu()

            for s_idx, s_feat in enumerate(feats.chunk(P.K_shift)):
                feats_all[s_idx] += [s_feat]

        for s_idx in range(P.K_shift):
            feats_all[s_idx] = torch.cat(feats_all[s_idx], dim=0)

    # compute clustering
    log_func("compute feature clusters")
    D0s = []
    for s_idx in range(P.K_shift):
        feats = feats_all[s_idx].numpy()
        clusters = get_feature_cluster(feats, P.sr_dict_size, log_func)
        D0 = torch.tensor(clusters.T)
        D0s.append(D0)
    D0s = torch.stack(D0s, dim=0)

    # set dictionary
    P.cocsr_net.sr_nets.set_dictionary(D0s, train_dict=True)

def get_feature_cluster(features, cluster_num, log_func):
    k_means = KMeans(init='k-means++', n_clusters=cluster_num, n_init=10)
    t0 = time.time()
    k_means.fit(features)
    t_batch = time.time() - t0

    log_func("clustering time: %.2fs" % t_batch)

    return k_means.cluster_centers_

