r"""
This file defines loss functions related to sparse dictionary learning and reconstruction

#TODO: refactor this 
"""

import torch
import torch.nn as nn

DICT_CONSTRAINT_EQ = 0
DICT_CONSTRAINT_LE = 1

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, D, X, S):
        r"""
        Args:
            D: [Feature size, Dictionary size], dictionary
            X: [Dictionary size, Data count], reconstruction coefficients
            S: [Feature size, Data count], original vector
    
        Returns:
            Tensor: sparse reconstruction loss
        """
        return compute_reconstruction_error(D, X, S)


class SparsityL1Loss(nn.Module):
    def __init__(self):
        super(SparsityL1Loss, self).__init__()

    def forward(self, X):
        r"""
        Args:
            X: [Dictionary size, Data count], reconstruction coefficients
    
        Returns:
            Tensor: sparse reconstruction loss
        """
        return compute_sparsity_l1(X)


class SparseReconstructionLossOption:
    def __init__(self, 
                 lambda_reconstruction, 
                 lambda_sparsity_l1, 
                 ) -> None:
        self.lambda_reconstruction = lambda_reconstruction

        self.lambda_sparsity_l1 = lambda_sparsity_l1

class SparseReconstructionLoss(nn.Module):
    r"""
    Compute sparse reconstruction loss
    """

    def __init__(self, 
                 lambda_reconstruction, 
                 lambda_sparsity_l1, 
                 ):
        super(SparseReconstructionLoss, self).__init__()
        
        self.lambda_reconstruction = lambda_reconstruction        
        self.reconstruction_loss_func = ReconstructionLoss()

        self.lambda_sparsity_l1 = lambda_sparsity_l1
        self.sparsity_l1_loss_func = SparsityL1Loss()

    @classmethod
    def from_option(cls, option: SparseReconstructionLossOption):
        return cls(lambda_reconstruction=option.lambda_reconstruction, 
                   lambda_sparsity_l1=option.lambda_sparsity_l1,
                   )

    def forward(self, D, X, S):
        r"""
        Args:
            D: [Feature size, Dictionary size], dictionary
            X: [Dictionary size, Data count], reconstruction coefficients
            S: [Feature size, Data count], original vector
    
        Returns:
            Tensor: sparse reconstruction loss
        """

        reconstruction_loss = self.reconstruction_loss_func(D, X, S)

        sparsity_l1_loss = self.sparsity_l1_loss_func(X)

        return (self.lambda_reconstruction * reconstruction_loss 
                + self.lambda_sparsity_l1 * sparsity_l1_loss
                )


class SparseDictionaryConstraintLossOption:
    def __init__(self, constr_mode=DICT_CONSTRAINT_EQ, constraint=1) -> None:
        self.constr_mode = constr_mode
        self.constraint = constraint

class SparseDictionaryConstraintLoss(nn.Module):
    def __init__(self, constr_mode=DICT_CONSTRAINT_EQ, constraint=1) -> None:
        super(SparseDictionaryConstraintLoss, self).__init__()

        self.constraint = constraint
        self.constr_mode = constr_mode

        if constr_mode == DICT_CONSTRAINT_EQ:
            self.dict_constr_loss_func = compute_dictionary_norm_constraint_loss_equal
        else: # dict_constr_mode == DICT_CONSTRAINT_LE:
            self.dict_constr_loss_func = compute_dictionary_norm_constraint_loss_less_equal

    @classmethod
    def from_option(cls, option: SparseDictionaryConstraintLossOption):
        return cls(constr_mode=option.constr_mode, 
                   constraint=option.constraint)    

    def forward(self, D):
        return self.dict_constr_loss_func(D, self.constraint)


class SparseDictionaryLearningLossOption:
    def __init__(self,
                 lambda_reconstruction, 
                 lambda_sparsity_l1, 
                 lambda_dict_constr, 
                 dict_constr_mode=DICT_CONSTRAINT_EQ, 
                 dict_constraint=1,
                 ) -> None:
        self.lambda_reconstruction = lambda_reconstruction

        self.lambda_sparsity_l1 = lambda_sparsity_l1

        self.lambda_dict_constr = lambda_dict_constr
        self.dict_constr_mode = dict_constr_mode
        self.dict_constraint = dict_constraint

class SparseDictionaryLearningLoss(nn.Module):
    r"""
    Compute sparse dictionary learning loss

    TODO: the way to compose this loss is a little weird
    """

    def __init__(self,  
                 lambda_reconstruction, 
                 lambda_sparsity_l1, 
                 lambda_dict_constr, 
                 dict_constr_mode=DICT_CONSTRAINT_EQ, 
                 dict_constraint=1,
                 ):
        super(SparseDictionaryLearningLoss, self).__init__()
        
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_sparsity_l1 = lambda_sparsity_l1
        self.sparse_reconstruction_loss_func = SparseReconstructionLoss(lambda_reconstruction, lambda_sparsity_l1)

        self.lambda_dict_constr = lambda_dict_constr
        self.sparse_dictionary_constraint_loss_func = SparseDictionaryConstraintLoss(constr_mode=dict_constr_mode, constraint=dict_constraint)

    def from_option(cls, option: SparseDictionaryLearningLossOption):
        return cls(lambda_reconstruction=option.lambda_reconstruction,
                   lambda_sparsity_l1=option.lambda_sparsity_l1,
                   lambda_dict_constr=option.lambda_dict_constr, 
                   dict_constr_mode=option.dict_constr_mode,
                   dict_constraint=option.dict_constraint
                   )

    def forward(self, D, X, S):
        r"""
        Args:
            D: [Feature size, Dictionary size], dictionary
            X: [Dictionary size, Data count], reconstruction coefficients
            S: [Feature size, Data count], original vector
    
        Returns:
            Tensor: sparse dicitonary learning loss
        """

        sparse_reconstruction_loss = self.sparse_reconstruction_loss_func(D, X, S)

        dictionary_constraint_loss = self.sparse_dictionary_constraint_loss_func(D)

        return sparse_reconstruction_loss + self.lambda_dict_constr * dictionary_constraint_loss

class SparseRepresentationClassificationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SparseRepresentationClassificationLoss, self).__init__()

        self.eps = eps

        self.softmax = nn.Softmax(dim=0)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, labels, Ds, Xs, S):
        r"""
        Args:
            labels: [Data number], class labels
            Ds: [Class number, Feature size, Dictionary size], dictionary
            Xs: [Class number, Dictionary size, Data count], reconstruction coefficients
            S: [Feature size, Data count], original vector

        Returns:
            Tensor: sparse representation classification loss
        """
        
        # DN x 1 x N, residuals of each dictionary of each data
        residuals = compute_reconstruction_error(Ds, Xs, S, keepdim=True).squeeze(1)

        # 1 x N, for each data, sum residual over all dictionary
        r_sum = residuals.sum(dim=0) 

        # pseudo class probability
        pseu_prob = (r_sum - residuals) / r_sum.clamp(min=self.eps) 

        prob = self.softmax(pseu_prob)

        c_loss = self.ce_loss(input=prob.t(), target=labels)
            
        return c_loss

class ConcatenatedDictionarySRCLoss(SparseRepresentationClassificationLoss):
    def __init__(self, dict_size, dict_num, eps=1e-8):
        super().__init__(eps=eps)

        self.dict_size = dict_size
        self.dict_num = dict_num

    def forward(self, labels, Ds, X_cat, S):
        r"""
        Args:
            labels: [Data number], class labels
            Ds: [Class number, Feature size, Dictionary size], separated dictionary
            X_cat: [1, Class number * Dictionary size, Data count], reconstruction coefficients computed with concatenated dictionary
            S: [Feature size, Data count], original vector

        Returns:
            Tensor: sparse representation classification loss
        """

        Xs = X_cat.view(self.dict_num, self.dict_size, -1)

        return super().forward(labels, Ds, Xs, S)

def compute_reconstruction(D, X):
    """
    Args:
        D: [*, Feature size, Dictionary size], dictionary
        X: [*, Dictionary size, Data count], reconstruction coefficients
    
    Returns:
        Tensor: Reconstructed vector
    """

    return D.matmul(X)

def compute_reconstruction_error(D, X, S, keepdim=False, mask=None):
    """
    Args:
        D: [*, Feature size, Dictionary size], dictionary
        X: [*, Dictionary size, Data count], reconstruction coefficients
        S: [Feature size, Data count], feature vectors to reconstruct
        mask: (optional) [*, Data count], indicate if the feature correspond to each dictionary
    
    Returns:
        Tensor: Errors between reconstructed vector and original vector
    """

    residual = compute_reconstruction(D, X) - S
    err = residual.norm(p=2, dim=-2, keepdim=True).square()

    if mask is not None:
        mask = mask.unsqueeze(-2)
        err = err * mask

    if keepdim:
        return err
    else:
        return err.sum()

def compute_sparsity_l1(X, keepdim=False, mask=None):
    """
    Args:
        X: [*, Dictionary size, Data count], reconstruction coefficients
        mask: (optional) [*, Data count], indicate if the feature correspond to each dictionary
    
    Returns:
        Tensor: sparsity of X
    """
    x_norm = X.norm(p=1, dim=-2, keepdim=True)

    if mask is not None:
        mask = mask.unsqueeze(-2)
        x_norm = x_norm * mask

    if keepdim:
        return x_norm
    else:
        return x_norm.sum()

def compute_dictionary_norm_constraint_loss_less_equal(D, constraint, keepdim=False):
    """
    Args:
        D: [*, Feature size, Dictionary size], dictionary
        constraint: 
    
    Returns:
        Tensor: overshoot of D higher than constraint
    """

    norm = D.norm(dim=-2, p=2, keepdim=keepdim)
    if keepdim:
        return (norm - constraint).clamp(min=0)
    else:
        return (norm - constraint).clamp(min=0).sum()

def compute_dictionary_norm_constraint_loss_equal(D, constraint, keepdim=False):
    """
    Args:
        D: [*, Feature size, Dictionary size], dictionary
        constraint: 
    
    Returns:
        Tensor: errors between D and constraint
    """

    norm = D.norm(dim=-2, p=2, keepdim=keepdim)
    if keepdim:
        (norm - constraint).abs()
    else:
        return (norm - constraint).abs().sum()

