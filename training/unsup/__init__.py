def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_unsup_{mode}'

    if mode == 'simclr':
        from .simclr import train
    elif mode == 'simclr_CSI':
        from .simclr_CSI import train
        fname += f'_shift_{P.shift_trans_type}'
    elif mode == 'cocsr_dict':
        from .cocsr_dict import train
        fname += f'_shift_{P.shift_trans_type}'
        fname += f'_cocsr_ld{P.sr_lambda}_lr{P.sr_lr}'
        if P.invert_sr_feature_norm:
            fname += '_invf'
    elif mode == 'bit':
        from .bit_oneclass_finetune import train
        fname += f'_bit'
        if P.pretrained_model:
            fname += '_pretrain'
    elif mode == 'swin':
        fname += f'_swin'
        if P.pretrained_model:
            fname += '_pretrain'
    else:
        raise NotImplementedError()

    if P.one_class_idx is not None:
        fname += f'_one_class_{P.one_class_idx}'

    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train, fname


def update_comp_loss(loss_dict, loss_in, loss_out, loss_diff, batch_size):
    loss_dict['pos'].update(loss_in, batch_size)
    loss_dict['neg'].update(loss_out, batch_size)
    loss_dict['diff'].update(loss_diff, batch_size)


def summary_comp_loss(logger, tag, loss_dict, epoch):
    logger.scalar_summary(f'{tag}/pos', loss_dict['pos'].average, epoch)
    logger.scalar_summary(f'{tag}/neg', loss_dict['neg'].average, epoch)
    logger.scalar_summary(f'{tag}', loss_dict['diff'].average, epoch)

