"""
This file define an one class classification network using sparse representation
"""

from typing import Optional, Tuple, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.admm_sr import AdmmSrNet

from utils.utils import normalize

__all__ = ['ContrastiveOneClassSrNet', 'ContrastiveOneClassSrNetOption']

class ContrastiveOneClassSrNetOption:
    def __init__(self, use_z_for_sr=False, 
                 invert_sr_feature_normsq=False,
                 normalize_sr_feature=True,
                 eps=1e-8) -> None:
        super().__init__()

        self.use_z_for_sr = use_z_for_sr

        self.invert_sr_feature_normsq = invert_sr_feature_normsq
        self.normalize_sr_feature = normalize_sr_feature

        self.eps = eps

class ContrastiveOneClassSrNet(nn.Module):
    def __init__(self, 
                 feature_net: BaseModel,
                 sr_nets: AdmmSrNet,
                 option: ContrastiveOneClassSrNetOption()) -> None:
        """
        Args:
            feature_net:
            sr_nets: multi dictionary SR network
            option:
        """
        
        super(ContrastiveOneClassSrNet, self).__init__()

        # TODO: check if the feature size is consistent
        self.feature_net = feature_net
        
        self.sr_nets = sr_nets        

        self.option = option

        pass

    def forward(self, input, 
                penultimate: bool = True, simclr: bool = True,        
                shift: bool = True,   
                joint: bool = False,      
                sr_feature: bool = True, sr_coef: bool = True,
                sr_ds: bool = False, sr_ds_stages: List[int] = [],
                detatch_feature = False,
                transpose_sr_coef = False,
                ) -> Dict[str, Tensor]:
        """
        Args:
            input: [data count, feature size]
            penultimate:
            simclr:
            joint: 
            shift:
            sr_feature:
            sr_coef:            
            sr_ds: 
            sr_ds_stages: 
            detatch_feature:
            transpose_sr_coef: 

        Return:
            output['classification']: [data count, class number], classificaiton output of the feature network (only valid for supervised learning)
            output['penultimate']: [data count, feature size], features before projection layer
            output['simclr']: [data count, feature size], features after projection layer
            output['shift']: [data count, augmentation class number], auxiliary classification of destructive augmentation 
            output['joint']: ?
            output['sr_feature']: [data count, feature size], features for sr
            output['sr_coef']: if transpose sr coef is true, [dictionary number][data count, dictionary size], 
                               otherwise, [dictionary number][dictionary size, data count], 
                               sparse reconstruction coefficients
            output['sr_ds']: if transpose sr coef is true, [dictionary number][sr deep supervision stage number][data count, dictionary size], 
                             otherwise, [dictionary number][sr deep supervision stage number][dictionary size, data count], 
            
        """ 

        _aux = {}
        _return_aux = False

        feature_net_return_aux = (penultimate 
                                or simclr
                                or shift
                                or joint
                                or sr_coef
                                )  

        # feature
        with torch.set_grad_enabled(not detatch_feature):
            if feature_net_return_aux:
                _return_aux = True

                f_penultimate = penultimate or (sr_coef and not self.option.use_z_for_sr)
                f_simclr = simclr or (sr_coef and self.option.use_z_for_sr)

                output, _aux = self.feature_net(input, f_penultimate, f_simclr, shift, joint)
            else:
                output = self.feature_net(input)
    
        if self.option.use_z_for_sr:
            sr_feature_vec = _aux['simclr'].t()
        else:
            sr_feature_vec = _aux['penultimate'].t()

        if self.option.invert_sr_feature_normsq:
            eps = 1e-8
            sr_feature_vec = sr_feature_vec / (sr_feature_vec.norm(dim=0, keepdim=True) + eps) ** 2

        if self.option.normalize_sr_feature:
            sr_feature_vec = normalize(sr_feature_vec, dim=0, eps=self.option.eps)

        if sr_feature:
            _return_aux = True
            _aux['sr_feature'] = sr_feature_vec.t()

        # sparse reconstruction
        compute_sr = sr_coef
        if compute_sr:               
            if sr_ds:
                x, x_ds = self.sr_nets(sr_feature_vec, sr_ds, sr_ds_stages)

                if transpose_sr_coef:
                    all_x = x.transpose(-1, -2)
                    all_x_ds = [x_i.transpose(-1, -2) for x_i in x_ds]
                else:
                    all_x = x
                    all_x_ds = x_ds
            else:
                x = self.sr_nets(sr_feature_vec)

                if transpose_sr_coef:
                    all_x = x.transpose(-1, -2)
                else:
                    all_x = x
            
            if sr_coef:
                _return_aux = True
                _aux['sr_coef'] = all_x

            if sr_ds:
                _return_aux = True
                _aux['sr_ds'] = all_x_ds
        

        if _return_aux:
            return output, _aux

        return output

    def update_cat_dict(self):
        # DN x F x D
        dicts_temp = self.sr_nets.get_dictionary().detach() 

        # DN x D x F
        dicts = dicts_temp.transpose(-1, -2)

        # 1 x (DN * D) x F
        dicts = dicts.reshape(1, -1, dicts_temp.size(-1))

        # 1 x F x (DN * D)
        dicts = dicts.transpose(-1, -2)

        self.cat_src_net.set_dictionary(dicts)

    
