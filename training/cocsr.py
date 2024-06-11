"""
"""

from typing import List

import torch
import torch.nn as nn

from training import contrastive
from training import sr

class CocsrLossRecord:        

    """
    TODO: there should be some better way to recored this
    """

    def __init__(self, total_loss=0, 
                 contrastive_loss=None,
                 feature_contrastive_sr_loss=None,
                 augmentation_classification_loss=None,
                 augmentation_sr_classification_loss=None,
                 augmentation_cat_sr_classification_loss=None,
                 batch_pair_contrastive_loss=None,

                 sr_reconstruction_loss=None, 
                 sr_l1_loss=None, 
                 sr_dict_constr_loss=None, 
                 sr_dict_contrastive_sr_loss=None,
                 ):
        self.total_loss = total_loss
        
        self.contrastive_loss = contrastive_loss
        self.feature_contrastive_sr_loss = feature_contrastive_sr_loss
        self.augmentation_classification_loss = augmentation_classification_loss
        self.augmentation_sr_classification_loss = augmentation_sr_classification_loss
        self.augmentation_cat_sr_classification_loss = augmentation_cat_sr_classification_loss
        self.batch_pair_contrastive_loss = batch_pair_contrastive_loss

        self.sr_reconstruction_loss = sr_reconstruction_loss
        self.sr_l1_loss = sr_l1_loss
        self.sr_dict_constr_loss = sr_dict_constr_loss
        self.sr_dict_contrastive_sr_loss = sr_dict_contrastive_sr_loss

    def __add__(self, other):
        c = self.__class__()

        if self.total_loss is not None and other.total_loss is not None:
            c.total_loss = self.total_loss + other.total_loss

        if self.contrastive_loss is not None and other.contrastive_loss is not None:
            c.contrastive_loss = self.contrastive_loss + other.contrastive_loss
        if self.feature_contrastive_sr_loss is not None and other.feature_contrastive_sr_loss is not None:
            c.feature_contrastive_sr_loss = self.feature_contrastive_sr_loss + other.feature_contrastive_sr_loss
        if self.augmentation_classification_loss is not None and other.augmentation_classification_loss is not None:
            c.augmentation_classification_loss = self.augmentation_classification_loss + other.augmentation_classification_loss
        if self.augmentation_sr_classification_loss is not None and other.augmentation_sr_classification_loss is not None:
            c.augmentation_sr_classification_loss = self.augmentation_sr_classification_loss + other.augmentation_sr_classification_loss
        if self.augmentation_cat_sr_classification_loss is not None and other.augmentation_cat_sr_classification_loss is not None:
            c.augmentation_cat_sr_classification_loss = self.augmentation_cat_sr_classification_loss + other.augmentation_cat_sr_classification_loss
        if self.batch_pair_contrastive_loss is not None and other.batch_pair_contrastive_loss is not None:
            c.batch_pair_contrastive_loss = self.batch_pair_contrastive_loss + other.batch_pair_contrastive_loss

        if self.sr_reconstruction_loss is not None and other.sr_reconstruction_loss is not None:
            c.sr_reconstruction_loss = self.sr_reconstruction_loss + other.sr_reconstruction_loss
        if self.sr_l1_loss is not None and other.sr_l1_loss is not None:
            c.sr_l1_loss = self.sr_l1_loss + other.sr_l1_loss
        if self.sr_dict_constr_loss is not None and other.sr_dict_constr_loss is not None:
            c.sr_dict_constr_loss = self.sr_dict_constr_loss + other.sr_dict_constr_loss
        if self.sr_dict_contrastive_sr_loss is not None and other.sr_dict_contrastive_sr_loss is not None:
            c.sr_dict_contrastive_sr_loss = self.sr_dict_contrastive_sr_loss + other.sr_dict_contrastive_sr_loss
        
        return c

    def __mul__(self, scale):
        c = self.__class__()

        if self.total_loss is not None: 
            c.total_loss = self.total_loss * scale

        if self.contrastive_loss is not None:
            c.contrastive_loss = self.contrastive_loss * scale
        if self.feature_contrastive_sr_loss is not None:
            c.feature_contrastive_sr_loss = self.feature_contrastive_sr_loss * scale
        if self.augmentation_classification_loss is not None:
            c.augmentation_classification_loss = self.augmentation_classification_loss * scale
        if self.augmentation_sr_classification_loss is not None:
            c.augmentation_sr_classification_loss = self.augmentation_sr_classification_loss * scale
        if self.augmentation_cat_sr_classification_loss is not None:
            c.augmentation_cat_sr_classification_loss = self.augmentation_cat_sr_classification_loss * scale
        if self.batch_pair_contrastive_loss is not None:
            c.batch_pair_contrastive_loss = self.batch_pair_contrastive_loss * scale

        if self.sr_reconstruction_loss is not None:
            c.sr_reconstruction_loss = self.sr_reconstruction_loss * scale
        if self.sr_l1_loss is not None:
            c.sr_l1_loss = self.sr_l1_loss  * scale
        if self.sr_dict_constr_loss is not None:
            c.sr_dict_constr_loss = self.sr_dict_constr_loss * scale
        if self.sr_dict_contrastive_sr_loss is not None:
            c.sr_dict_contrastive_sr_loss = self.sr_dict_contrastive_sr_loss * scale
        
        return c

class BaseContrastiveSRLoss(nn.Module):
    """
    Base class of contrastive sr loss using similarity between features and dictionary
    """

    def __init__(self, normalize=False, tau=1, eps=1e-8):
        super(BaseContrastiveSRLoss, self).__init__()
        self.normalize = normalize
        self.tau = tau
        self.epsilon = eps

    @classmethod
    def from_option(cls, option: contrastive.ContrastiveLossOption):
        return cls(option.normalize, option.tau, option.eps)

    def _compute_similarity(self, dict, alpha, feature, normalize):
        """
        Compute similarity to the dictoinary
        need this?
    
        Args:
            dict: [feature dimension, dictionary size], sparse dictionary
            alpha: [dictionary size, data number], sparse reconstruction coefficient
            feature: [feature dimension, data number], vector to reconstruct

        Returns:
            Tensor: [data number], feature similarity to the dictionary 
        """
        f_r = sr.compute_reconstruction(D=dict, X=alpha) # D x N
        f_sim = (f_r * feature).sum(dim=0) # N

        if normalize:
            sim_norm = (f_r.norm(dim=0) * feature.norm(dim=0)).clamp(min=self.epsilon)
            f_sim = f_sim / sim_norm

        return f_sim
  
class DictionaryContrastiveSRLoss(BaseContrastiveSRLoss):
    """
    Contrastive sr loss using similarity between features and dictionary
    dictionary major
    """

    def __init__(self, normalize=False, tau=1, eps=1e-8):
        super(DictionaryContrastiveSRLoss, self).__init__(normalize=normalize, tau=tau, eps=eps)
        
    @classmethod
    def from_option(cls, option: contrastive.ContrastiveLossOption):
        return cls(option.normalize, option.tau, option.eps)
    
    def forward(self, dicts, alphas, feature, positive_indices):
        """
        Args:
            dicts: [dictionary number][feature dimension, dictionary size], sparse dictionary
            alphas: [dictionary number][dictionary size, data number], sparse reconstruction coefficient
            feature: [feature dimension, data number], vector to reconstruct
            positive_indices: [dictionary number][postive set data number], index of data in the positive set
        """
        
        raise NotImplementedError()

        # sum_i (sim(D_i, f_i) / (sum_j(sim(D_i, f_j))))
        total_loss = 0
        for dict_i, _ in enumerate(dicts):  
            if len(positive_indices[dict_i]) == 0:
                continue

            sim_all = self._compute_similarity(dicts[dict_i], alphas[dict_i], feature, self.normalize) # N
            
            # compute loss term
            sim_all = torch.exp(sim_all / self.tau)
            sim_pos = sim_all[positive_indices[dict_i]]

            dict_loss = -torch.log(sim_pos.sum() / sim_all.sum().clamp(min=self.epsilon)).mean()

            total_loss = total_loss + dict_loss

        return total_loss

class FeatureContrastiveSRLoss(BaseContrastiveSRLoss):
    """
    Contrastive sr loss using similarity between features and dictionary
    features major
    """

    def __init__(self, normalize=False, tau=1, eps=1e-8):
        super(FeatureContrastiveSRLoss, self).__init__(normalize=normalize, tau=tau, eps=eps)
        
    @classmethod
    def from_option(cls, option: contrastive.ContrastiveLossOption):
        return cls(option.normalize, option.tau, option.eps)
    
    def forward(self, dicts, alphas, feature, positive_indices):
        """
        Args:
            dicts: [dictionary number][feature dimension, dictionary size], sparse dictionary
            alphas: [dictionary number][dictionary size, data number], sparse reconstruction coefficient
            feature: [feature dimension, data number], vector to reconstruct
            positive_indices: [dictionary number][postive set data number], index of data in the positive set
        """

        raise NotImplementedError()

        # feature similarity to all dictionaries
        dict_sim = []
        for dict_i, _ in enumerate(dicts):
            sim_all = self._compute_similarity(dicts[dict_i], alphas[dict_i], feature, self.normalize) # N
            sim_all = torch.exp(sim_all / self.tau)
            dict_sim.append(sim_all)

        # sum_i (sim(D_i, f_i) / (sum_j(sim(D_j, f_i))))        
        total_loss = 0
        for dict_i, _ in enumerate(dicts):
            if len(positive_indices[dict_i]) == 0:
                continue
            
            sim_all_sum = 0
            
            for dict_j, in enumerate(dicts):        
                sim_all_sum = sim_all_sum + dict_sim[dict_j][positive_indices[dict_i]].sum()
            
            # compute loss term            
            sim_pos_sum = dict_sim[dict_i][positive_indices[dict_i]].sum()

            dict_loss = -torch.log(sim_pos_sum / sim_all_sum.clamp(min=self.epsilon)).mean()

            total_loss = total_loss + dict_loss

        return total_loss

class CocsrFeatureLearningLossOption():
    def __init__(self, 
                 compute_contrastive_loss=True, 
                 loss_lambda_contrastive=1,
                 contrastive_loss_op: contrastive.ContrastiveLossOption = contrastive.ContrastiveLossOption(),
                 use_h_for_contrastive=False,

                 compute_fea_contra_sr_loss=False,
                 loss_lambda_fea_contra_sr=1,
                 fea_contra_sr_loss_op: contrastive.ContrastiveLossOption = contrastive.ContrastiveLossOption(),
                 use_z_for_sr=False,    

                 compute_aug_cls_loss=True,
                 loss_lambda_aug_cls=1,

                 compute_aug_src_loss=False,
                 loss_lambda_aug_src=1,

                 compute_aug_cat_src_loss=False,
                 loss_lambda_aug_cat_src=1,
                 cat_dict_size=512,
                 cat_dict_number=1,

                 compute_batch_pair_contrastive_loss=False,
                 loss_lambda_batch_pair_contrastive=0,
                 batch_pair_contrastive_loss_op: contrastive.BatchPairContrastiveLossOption = contrastive.BatchPairContrastiveLossOption(),                 
                 
                 verbose=False,
                 debug=False
                 ) -> None:
        self.compute_contrastive_loss = compute_contrastive_loss
        self.loss_lambda_contrastive = loss_lambda_contrastive
        self.contrastive_loss_op = contrastive_loss_op
        self.use_h_for_contrastive = use_h_for_contrastive

        self.compute_fea_contra_sr_loss = compute_fea_contra_sr_loss
        self.loss_lambda_fea_contra_sr = loss_lambda_fea_contra_sr
        self.fea_contra_sr_loss_op = fea_contra_sr_loss_op                  
        self.use_z_for_sr = use_z_for_sr

        self.compute_aug_cls_loss = compute_aug_cls_loss
        self.loss_lambda_aug_cls = loss_lambda_aug_cls
        
        self.compute_aug_src_loss = compute_aug_src_loss
        self.loss_lambda_aug_src = loss_lambda_aug_src
        
        self.compute_aug_cat_src_loss = compute_aug_cat_src_loss
        self.loss_lambda_aug_cat_src = loss_lambda_aug_cat_src
        self.cat_dict_size = cat_dict_size
        self.cat_dict_number = cat_dict_number
        
        self.compute_batch_pair_contrastive_loss = compute_batch_pair_contrastive_loss
        self.loss_lambda_batch_pair_contrastive = loss_lambda_batch_pair_contrastive
        self.batch_pair_contrastive_loss_op = batch_pair_contrastive_loss_op

        self.verbose = verbose
        self.debug = debug

class CocsrFeatureLearningLoss(nn.Module):
    """
    Loss for feature training stage
    """

    def __init__(self, 
                 option: CocsrFeatureLearningLossOption = CocsrFeatureLearningLossOption()
                 ):
        super(CocsrFeatureLearningLoss, self).__init__()
                
        self.opt = option

        self.contrastive_loss_func = contrastive.ContrastiveLoss.from_option(option.contrastive_loss_op)
        
        self.fea_contra_sr_loss_func = FeatureContrastiveSRLoss.from_option(option.fea_contra_sr_loss_op)

        self.aug_cls_loss_func = nn.CrossEntropyLoss()

        self.aug_src_loss_func = sr.SparseRepresentationClassificationLoss()        
        self.aug_cat_src_loss_func = sr.ConcatenatedDictionarySRCLoss(dict_size=option.cat_dict_size, dict_num=option.cat_dict_number)

        self.batch_pair_contrastive_loss_func = contrastive.BatchPairContrastiveLoss.from_option(option.batch_pair_contrastive_loss_op)

    @classmethod
    def from_option(cls, option: CocsrFeatureLearningLossOption):
        return cls(option)

    def forward(self, 
                aug_labels,
                dict_labels,
                z_i, z_j, 
                h_i, h_j,
                s_i, s_j,
                all_x_i: List[torch.Tensor], all_x_j: List[torch.Tensor],
                cat_x_i, cat_x_j,
                all_dict: List[torch.Tensor]
                ):        
        r"""
        Args:          
            aug_labels: [data count]
            labels of destructive augmentation type 

            dict_index: [data count]
            dictionary index of each data 

            z_i: [data count, feature size] 
            z_j: [data count, feature size]  
            Contrastive feature projection           
            
            h_i: [data count, feature size] 
            h_j: [data count, feature size]      
            Contrastive feature
            
            s_i: [data count, destructive augmentation type number] 
            s_j: [data count, destructive augmentation type number]      
            destructive augmentation classification prediction

            all_x_i: [dictionary number][dictionary size, data count]
            all_x_j: [dictionary number][dictionary size, data count]
            Sparse reconstruction coefficients
            
            cat_x_i:
            cat_x_j:
            
            all_dict: [dictionary number][feature_length, dictionary_size]
            Sparse dictionary

        Returns: 
            Tensor: contrastive loss
            [Tensor]: list of sub loss if verbose
        """     

        loss_for_learning = 0  
        loss_record = CocsrLossRecord()

        # feature contrastive loss
        if (self.opt.debug 
            or self.opt.compute_contrastive_loss
            or self.opt.compute_batch_pair_contrastive_loss):
            if self.opt.use_h_for_contrastive:
                z_contra_i = h_i
                z_contra_j = h_j
            else:
                z_contra_i = z_i
                z_contra_j = z_j
            
            if (self.opt.debug or self.opt.compute_contrastive_loss):        
                contrastive_loss = self.contrastive_loss_func(z_contra_i, z_contra_j)
                
                loss_record.contrastive_loss = contrastive_loss.item()

                if self.opt.compute_contrastive_loss:
                    loss_for_learning = loss_for_learning + self.opt.loss_lambda_contrastive * contrastive_loss

            if (self.opt.debug or self.opt.compute_batch_pair_contrastive_loss):
                batch_pair_contrastive_loss = self.batch_pair_contrastive_loss_func(xi=z_contra_i, xj=z_contra_j, labels=aug_labels)
                
                loss_record.batch_pair_contrastive_loss = batch_pair_contrastive_loss.item()

                if self.opt.compute_batch_pair_contrastive_loss:
                    loss_for_learning = loss_for_learning + self.opt.loss_lambda_batch_pair_contrastive * batch_pair_contrastive_loss

        # sr related loss
        if (self.opt.debug 
            or self.opt.compute_fea_contra_sr_loss
            or self.opt.compute_aug_src_loss
            or self.opt.compute_aug_cat_src_loss):
            
            dict_num = len(all_dict)
            dict_pos_idx = [[]] * dict_num
            for dict_index in range(dict_num):
                pos_idx = [i for i, x in enumerate(dict_labels) if x == dict_index]
                dict_pos_idx[dict_index] = pos_idx
            
            if self.opt.use_z_for_sr:
                feature_i = z_i.t()
                feature_j = z_j.t()
            else:
                feature_i = h_i.t() 
                feature_j = h_j.t()

            if self.opt.debug or self.opt.compute_fea_contra_sr_loss:
                fea_contra_sr_loss_i = self.fea_contra_sr_loss_func(dicts=all_dict, alphas=all_x_i, feature=feature_i, positive_indices=dict_pos_idx)
                fea_contra_sr_loss_j = self.fea_contra_sr_loss_func(dicts=all_dict, alphas=all_x_j, feature=feature_j, positive_indices=dict_pos_idx)

                fea_contra_sr_loss = fea_contra_sr_loss_i + fea_contra_sr_loss_j

                loss_record.feature_contrastive_sr_loss = fea_contra_sr_loss.item()
            
                if self.opt.compute_fea_contra_sr_loss:
                    loss_for_learning = loss_for_learning + self.opt.loss_lambda_fea_contra_sr * fea_contra_sr_loss

            if self.opt.debug or self.opt.compute_aug_src_loss:                
                aug_src_loss_i = self.aug_src_loss_func(labels=aug_labels, Ds=all_dict, Xs=all_x_i, S=feature_i)
                aug_src_loss_j = self.aug_src_loss_func(labels=aug_labels, Ds=all_dict, Xs=all_x_j, S=feature_j)

                aug_src_loss = aug_src_loss_i + aug_src_loss_j
                
                loss_record.augmentation_sr_classification_loss = aug_src_loss.item()

                if self.opt.compute_aug_src_loss:
                    loss_for_learning = loss_for_learning + self.opt.loss_lambda_aug_src * aug_src_loss

            if self.opt.debug or self.opt.compute_aug_cat_src_loss:                
                aug_cat_src_loss_i = self.aug_cat_src_loss_func(labels=aug_labels, Ds=all_dict, X_cat=cat_x_i, S=feature_i)
                aug_cat_src_loss_j = self.aug_cat_src_loss_func(labels=aug_labels, Ds=all_dict, X_cat=cat_x_j, S=feature_j)

                aug_cat_src_loss = aug_cat_src_loss_i + aug_cat_src_loss_j
                
                loss_record.augmentation_cat_sr_classification_loss = aug_cat_src_loss.item()

                if self.opt.compute_aug_src_loss:
                    loss_for_learning = loss_for_learning + self.opt.loss_lambda_aug_cat_src * aug_cat_src_loss

        # auxiliary classifier loss
        if self.opt.debug or self.opt.compute_aug_cls_loss:

            aug_labels_cat = aug_labels.repeat(2).type(torch.cuda.LongTensor) # TODO: device
            aug_pred = torch.cat([s_i, s_j], 0)
            aug_cls_loss = self.aug_cls_loss_func(aug_pred, aug_labels_cat)
            
            loss_record.augmentation_classification_loss = aug_cls_loss.item()

            if self.opt.compute_aug_cls_loss:
                loss_for_learning = loss_for_learning + self.opt.loss_lambda_aug_cls * aug_cls_loss
      
        # make it tensor if no loss is calculated to prevent crash...
        if not isinstance(loss_for_learning, torch.Tensor):
            loss_for_learning = torch.zeros(1)

        if self.opt.verbose or self.opt.debug:
            return loss_for_learning, loss_record
        else:
            return loss_for_learning

class CocsrDictionaryLearningLossOption:
    """
    TODO: clean up and refactor this
    """

    def __init__(self, 
                 compute_sr_loss=True, 
                 loss_lambda_sr=1,
                 detach_sr_coef_for_sr=False,

                 compute_sr_l1_loss=True, 
                 loss_lambda_sr_l1=0.01,
                 
                 compute_dict_constr_loss=True,
                 loss_lambda_dict_constr=0.1,
                 dict_constr_loss_op: sr.SparseDictionaryConstraintLossOption = sr.SparseDictionaryConstraintLossOption(),
                                  
                 compute_sr_deep_supervision=False,
                                
                 verbose=False,
                 debug=False
                 ):
                
        self.compute_sr_loss = compute_sr_loss
        self.loss_lambda_sr = loss_lambda_sr
        self.detach_sr_coef_for_sr = detach_sr_coef_for_sr
        
        self.compute_sr_l1_loss = compute_sr_l1_loss
        self.loss_lambda_sr_l1 = loss_lambda_sr_l1
        
        self.compute_dict_constr_loss = compute_dict_constr_loss
        self.loss_lambda_dict_constr = loss_lambda_dict_constr
        self.dict_constr_loss_op = dict_constr_loss_op
                          
        self.compute_sr_deep_supervision = compute_sr_deep_supervision

        self.verbose = verbose
        self.debug = debug

class CocsrDictionaryLearningLoss(nn.Module):
    """
    Loss for ditionary learning and update stage

    TODO: use loss class in sr. 
    """

    def __init__(self,                 
                 option: CocsrDictionaryLearningLossOption = CocsrDictionaryLearningLossOption()
                 ):
        super(CocsrDictionaryLearningLoss, self).__init__()

        self.opt = option               

        self.sr_loss_func = sr.compute_reconstruction_error

        self.sr_l1_loss_func = sr.compute_sparsity_l1

        self.dict_constr_loss_func = sr.SparseDictionaryConstraintLoss.from_option(option.dict_constr_loss_op)
                
    @classmethod
    def from_option(cls, option: CocsrDictionaryLearningLossOption):
        return cls(option)

    def forward(self, 
                dict_labels,
                feature: torch.Tensor, 
                all_x: torch.Tensor,
                all_x_ds: List[torch.Tensor],
                all_dict: torch.Tensor):        
        r"""
        Args:
            dict_labels: [data count]
            dictionary index of each data 

            feature: [data count, feature size]            
            SR feature
            
            all_x: [dictionary number, dictionary size, data count]
            Sparse reconstruction coefficients
            
            all_x_ds: [sr deep supervision stage number][dictionary number, dictionary size, data count]
            Sparse reconstruction coefficients from intermediate stages

            all_dict: [dictionary number, feature_length, dictionary_size]
            Sparse dictionary
        """
                
        _device = feature.device

        dict_num = all_dict.size(0)
        data_count = all_x.size(-1)
        sr_ds_stage_num = len(all_x_ds) if all_x_ds is not None else 0

        loss_for_learning = 0
        loss_record = CocsrLossRecord()
       
        feature = feature.t()    

        # positive elements of each dictionary
        dict_mask = torch.zeros(size=(dict_num, data_count), device=_device, dtype=torch.int)
        dict_pos_idx = [[] for _ in range(dict_num)] 
        for dict_index in range(dict_num):            
            pos_idx = [i for i, x in enumerate(dict_labels) if x == dict_index]            
            
            dict_mask[dict_index][pos_idx] = 1
            dict_pos_idx[dict_index] = pos_idx

        compute_sr_deep_supervision = self.opt.compute_sr_deep_supervision and sr_ds_stage_num > 0
        
        # sr l1 loss requires x gradient, keep the original x object 
        x_d = all_x.detach()
        x_ds_d = [x_ds_st.detatch() for x_ds_st in all_x_ds]

        # reconstruction loss
        if self.opt.debug or self.opt.compute_sr_loss:
            if self.opt.detach_sr_coef_for_sr:
                sr_loss = self.sr_loss_func(all_dict, x_d, feature, mask=dict_mask)
            else:
                sr_loss = self.sr_loss_func(all_dict, all_x, feature, mask=dict_mask)

            if compute_sr_deep_supervision:
                sr_ds_loss = torch.zeros_like(sr_loss)

                x_ds_for_sr = []
                if self.opt.detach_sr_coef_for_sr:
                    x_ds_for_sr = x_ds_d                        
                else:
                    x_ds_for_sr = all_x_ds
                        
                for x_ds_st in x_ds_for_sr:
                    sr_ds_loss = sr_ds_loss + self.sr_loss_func(all_dict, x_ds_st, feature, mask=dict_mask)
                #sr_ds_loss = sr_ds_loss / sr_ds_stage_num
                sr_loss = sr_loss + sr_ds_loss

            if loss_record.sr_reconstruction_loss is None:
                loss_record.sr_reconstruction_loss = sr_loss.item()
            else:
                loss_record.sr_reconstruction_loss += sr_loss.item()

            if self.opt.compute_sr_loss:
                loss_for_learning = loss_for_learning + self.opt.loss_lambda_sr * sr_loss
        
        # sparsity loss
        if self.opt.debug or self.opt.compute_sr_l1_loss:
            sr_l1_loss = self.sr_l1_loss_func(all_x, mask=dict_mask)

            if compute_sr_deep_supervision:
                sr_ds_l1_loss = torch.zeros_like(sr_l1_loss)
                for x_ds_st in all_x_ds:
                    sr_ds_l1_loss = sr_ds_l1_loss + self.sr_l1_loss_func(x_ds_st, mask=dict_mask)
                #sr_ds_l1_loss = sr_ds_l1_loss / sr_ds_stage_num
                sr_l1_loss = sr_l1_loss + sr_ds_l1_loss

            if loss_record.sr_l1_loss is None:
                loss_record.sr_l1_loss = sr_l1_loss.item()
            else:
                loss_record.sr_l1_loss += sr_l1_loss.item()

            if self.opt.compute_sr_l1_loss:
                loss_for_learning = loss_for_learning + self.opt.loss_lambda_sr_l1 * sr_l1_loss

        # dictionary constraint loss
        if self.opt.debug or self.opt.compute_dict_constr_loss:
            dict_constr_loss = self.dict_constr_loss_func(all_dict)
            
            if loss_record.sr_dict_constr_loss is None:
                loss_record.sr_dict_constr_loss = dict_constr_loss.item()
            else:
                loss_record.sr_dict_constr_loss += dict_constr_loss.item()

            if self.opt.compute_dict_constr_loss:
                loss_for_learning = loss_for_learning + self.opt.loss_lambda_dict_constr * dict_constr_loss       

        # make it tensor if no loss is calculated to prevent crash...
        if not isinstance(loss_for_learning, torch.Tensor):
            loss_for_learning = torch.zeros(1)

        if self.opt.verbose or self.opt.debug:
            return loss_for_learning, loss_record
        else:
            return loss_for_learning
