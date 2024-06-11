r"""
This file defines contrastive loss
"""

import torch
import torch.nn as nn

class ContrastiveLossOption:
    """
    Args:
        normalize:
        tau:
        eps: min value of denominator to prevent #DIV/0! error 
        pos_in_denom: if denominator include positive pairs

        verbose: enable verbose mode
        debug: enable debug mode
    """
    def __init__(self, normalize=True, tau=0.5, eps=1e-8, pos_in_denom=False, verbose=False, debug=False) -> None:
        self.normalize = normalize
        self.tau = tau
        self.eps = eps
        self.pos_in_denom = pos_in_denom

        self.verbose = verbose
        self.debug = debug

class ContrastiveLoss(nn.Module):    
    def __init__(self, opt: ContrastiveLossOption = ContrastiveLossOption()):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt

    @classmethod
    def from_option(cls, option: ContrastiveLossOption):
        return cls(option)

    def forward(self, x_i, x_j):
        r"""
        Args: 
            xi: [Batch size, Feature size], features of an image augmentation
            xj: [Batch size, Feature size], features of another image augmentation

        TODO: sometimes the features become all 0, may need to do something with this
        """

        x = torch.cat((x_i, x_j), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.opt.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=self.opt.eps)

        sim_mat = torch.exp(sim_mat / self.opt.tau)

        # top
        if self.opt.normalize:
            sim_mat_denom = torch.norm(x_i, dim=1) * torch.norm(x_j, dim=1)
            sim_match = torch.exp(torch.sum(x_i * x_j, dim=-1) / sim_mat_denom.clamp(min=self.opt.eps) / self.opt.tau)
        else:
            sim_match = torch.exp(torch.sum(x_i * x_j, dim=-1) / self.opt.tau)
        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.opt.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum

        if self.opt.pos_in_denom:
            loss_denom = (torch.sum(sim_mat, dim=-1) - norm_sum)
        else:
            loss_denom = (torch.sum(sim_mat, dim=-1) - sim_match - norm_sum)

        loss_intermediate = -torch.log(sim_match / loss_denom.clamp(min=self.opt.eps))
        loss = torch.mean(loss_intermediate)

        # for debugging 
        if self.opt.debug and torch.isnan(loss):
            print("contrastive loss NAN")
            nan_loss_idx = [i for i, v in enumerate(loss_intermediate) if torch.isnan(v)]
            print(nan_loss_idx)

        return loss

class BatchPairContrastiveLossOption(ContrastiveLossOption):
    def __init__(self, 
                 device="cuda",
                 w_p_in=1, w_p_pooc=0.5,
                 w_n_in=0, w_n_pooc=0.5, w_n_des=1, w_n_mix=1,
                 normalize=True, tau=0.5, 
                 eps=1e-8, 
                 pos_in_denom=False, 
                 verbose=False, debug=False) -> None:
        super(BatchPairContrastiveLossOption, self).__init__(normalize, tau, eps, pos_in_denom, verbose, debug)

        self.device = device

        self.w_p_in = w_p_in
        self.w_p_pooc = w_p_pooc

        self.w_n_in = w_n_in
        self.w_n_pooc = w_n_pooc
        self.w_n_des = w_n_des
        self.w_n_mix = w_n_mix

class BatchPairContrastiveLoss(nn.Module):
    def __init__(self, opt: BatchPairContrastiveLossOption):
        super(BatchPairContrastiveLoss, self).__init__()
        self.opt = opt

    @classmethod
    def from_option(cls, option: BatchPairContrastiveLossOption):
        return cls(option)

    def forward(self, xi, xj, labels):

        ### utilities 
        batch_size = labels.size()[0]
        eye_B = torch.eye(batch_size).to(self.device)
        eye_2B = torch.eye(batch_size * 2).to(self.device)

        x = torch.cat((xi, xj), dim=0)

        labels_2 = torch.cat([labels, labels], dim=0).unsqueeze(1)
        
        ### mask of different type of pair
        
        # mix > destructive > pooc 
        pair_type = torch.max(labels_2, labels_2.t())

        # merge the same type 
        # TODO: make this not hard coded
        pair_type[pair_type == 2] = 1
        pair_type[pair_type == 3] = 1
        pair_type[pair_type == 4] = 1
        pair_type[pair_type == 5] = 1
        pair_type[pair_type == 6] = 1

        # pair of the same sample
        self_pair_mask = eye_2B

        # pair of two augmentation from the same image
        self_contra_pair_mask = torch.zeros_like(pair_type)
        self_contra_pair_mask[:batch_size, batch_size:] = eye_B
        self_contra_pair_mask[batch_size:, :batch_size] = eye_B

        # pair of two inclass sample
        inclass_sample_pair_mask = torch.zeros_like(pair_type)
        inclass_sample_pair_mask[pair_type == 0] = 1
        inclass_sample_pair_mask = inclass_sample_pair_mask * (1 - self_pair_mask)

        # pair of two augmentation from the same inclass sample
        inclass_pair_mask = inclass_sample_pair_mask * self_contra_pair_mask
        
        # pair of two different inclass sample
        pooc_pair_mask = inclass_sample_pair_mask * (1 - self_contra_pair_mask)
        
        # pair contains destructive samples 
        destructive_pair_mask = torch.zeros_like(pair_type) 
        destructive_pair_mask[pair_type == 1] = 1
        destructive_pair_mask = destructive_pair_mask * (1 - self_pair_mask) * (1 - self_contra_pair_mask)

        # pair contains mixup samples 
        mix_pair_mask = torch.zeros_like(pair_type) 
        mix_pair_mask[pair_type == 7] = 1
        mix_pair_mask = mix_pair_mask * (1 - self_pair_mask) * (1 - self_contra_pair_mask)

        ### similarity matrix
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=self.eps)
        sim_mat = torch.exp(sim_mat / self.tau) * (1 - self_pair_mask) 

        inclass_sim = sim_mat * inclass_pair_mask
        pooc_sim = sim_mat * pooc_pair_mask
        des_sim = sim_mat * destructive_pair_mask
        mix_sim = sim_mat * mix_pair_mask

        ### loss
        loss_no = (torch.sum(inclass_sim) * self.w_p_in 
                  + torch.sum(pooc_sim) * self.w_p_pooc)
        loss_deno = (torch.sum(inclass_sim) * self.w_n_in 
                    + torch.sum(pooc_sim) * self.w_n_pooc
                    + torch.sum(des_sim) * self.w_n_des
                    + torch.sum(mix_sim) * self.w_n_mix)

        loss = torch.mean(-torch.log(loss_no/ loss_deno.clamp(min=self.eps)))
        
        return loss