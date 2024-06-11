from common.eval import *

import os

import numpy as np

model.eval()

if P.mode == 'test_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == 'test_marginalized_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

elif P.mode in ['ood_pre', 
                'ood_pre_refmix', 'ood_pre_flipmix', 
                'ood_pre_cocsr', 'ood_pre_cocsr_refmix', 
                'ood_pre_multi_sr', 'ood_pre_multi_sr_refmix',
                'ood_pre_sr', 
                'ood_pre_tsne',]:
    if P.mode == 'ood':
        from evals import eval_ood_detection
    elif P.mode == 'ood_pre':
        from evals.ood_pre import eval_ood_detection
    elif P.mode == 'ood_pre_refmix':
        from evals.ood_pre_refmix import eval_ood_detection
    elif P.mode == 'ood_pre_flipmix':
        from evals.ood_pre_flipmix import eval_ood_detection
    elif P.mode == 'ood_pre_svdd':
        from evals.ood_pre_multi_svdd import eval_ood_detection
    elif P.mode == 'ood_pre_cocsr':
        from evals.ood_pre_cocsr import eval_ood_detection
    elif P.mode == 'ood_pre_cocsr_refmix':
        from evals.ood_pre_cocsr_refmix import eval_ood_detection
    elif P.mode == 'ood_pre_sr':
        from evals.ood_pre_sr import eval_ood_detection
    elif P.mode == 'ood_pre_multi_sr':
        from evals.ood_pre_multi_sr import eval_ood_detection
    else: # P.mode == 'ood_pre_multi_sr_refmix'
        from evals.ood_pre_multi_sr_refmix import eval_ood_detection

    with torch.no_grad():
        auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                        train_loader=train_loader, simclr_aug=simclr_aug, simclr_aug_no_crop=simclr_aug_no_crop)

    if P.one_class_idx is not None:
        mean_dict = dict()
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
        auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
        if P.print_score:
            print(message)
        bests.append(best_auroc)

    bests_it = map('{:.4f}'.format, bests)
    print('\t'.join(bests_it))

    if P.save_score:
        base_path = os.path.split(P.load_path)[0]  # checkpoint directory

        prefix = P.mode + '_' + f'{P.ood_samples}'
        
        if P.resize_fix:
            prefix += f'_resize_fix_{P.resize_factor}'
        else:
            prefix += f'_resize_range_{P.resize_factor}'

        if P.mode == 'ood_pre_multi_sr' or P.mode == 'ood_pre_multi_sr_refmix':
            prefix += f'_dict_{P.sr_dict_size}_ld_{P.sr_lambda}'

        score_prefix = os.path.join(base_path, f'{P.mode}_score_{prefix}')

        score_path = score_prefix + f'.txt'

        np.savetxt(score_path, bests, "%.4f")

else:
    raise NotImplementedError()


