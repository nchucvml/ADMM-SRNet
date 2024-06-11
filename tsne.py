from common.eval import *

model.eval()

if P.mode == 'ood_pre_tsne_sr':
    from evals.ood_pre_tsne_sr import eval_ood_tsne
elif P.mode == 'ood_pre_tsne_dict':
    from evals.ood_pre_tsne_dict import eval_ood_tsne
elif P.mode == 'ood_pre_tsne_dict_refmix':
    from evals.ood_pre_tsne_dict_refmix import eval_ood_tsne
elif P.mode == 'ood_pre_tsne_shi':
    from evals.ood_pre_tsne_shi import eval_ood_tsne
else:
    from evals.ood_pre_tsne import eval_ood_tsne

eval_ood_tsne(P, model, test_loader, ood_test_loader, P.ood_score,
              train_loader=train_loader, simclr_aug=simclr_aug)




