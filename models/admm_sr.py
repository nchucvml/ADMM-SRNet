"""
Network that simulate ADMM process to solve sparse representation problems
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['AdmmSrNet']

class AdmmSrNet(nn.Module):
    """
    Network simulating ADMM process to solve sparse representation problem
    $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$    
    """

    def __init__(self, 
                dictionary_number: int, feature_size: int, dictionary_size: int, 
                stage_number: int, 
                initial_dict: Tensor = None, train_dict = True,
                initial_rho: float = 1.0, train_rho = False,
                initial_lambda: float = 0.1, train_lambda = False,
                dict_constraint: float = 1.0,
                ) -> None:
        if (initial_dict is not None 
                and initial_dict.shape is not (feature_size, dictionary_size)):
            raise ValueError("initial_dict dimension is inconsistent to (feature_size, dictionary_size)")

        super(AdmmSrNet, self).__init__()

        self.dictionary_number = dictionary_number
        self.feature_size = feature_size 
        self.dictionary_size = dictionary_size
        self.dict_constraint = dict_constraint
        self.stage_number = stage_number

        if initial_dict is None:
            d0 = torch.rand((dictionary_number, feature_size, dictionary_size), 
                            dtype=torch.float)
        else:
            # TODO: can we use tensor without copying
            d0 = torch.tensor(initial_dict)
        self._set_dictionary(d0, train_dict)
        
        # Learnable rho and lambda may better fit the data.
        # However, they may become negative which violate the constraint
        # TODO: add positive constraint for learnable rho and lambda
        self.rho = nn.Parameter(torch.tensor(initial_rho), requires_grad=train_rho)
        self.lambdaV = nn.Parameter(torch.tensor(initial_lambda), requires_grad=train_lambda)

        self.admm_iteration = nn.ModuleList()
        for _ in range(0, stage_number):
            self.admm_iteration.append(AdmmBlock())

    def forward(self, input_data, return_sr_ds=False, deep_supervision_stages=[]):
        """
        ADMM process to solve sparse representation problem
        $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$
        
        ND: dictionary number
        M: feature size
        N: dictionary size
        K: input number

        input_data: S, M x K 2D tensor
        
        self.dictionary: D, ND x M x N 3D tensor
        X, Y, U: ND x N x K 3D tensor

        Returns:
            Tensor: ND x N x K
        """

        M = self.feature_size
        N = self.dictionary_size
        K = input_data.shape[-1]
        ND = self.dictionary.size(0)
        
        D = self.dictionary
        S = input_data

        DTS = torch.matmul(D.transpose(-1, -2), S)
        DTDIc = self._cholesky_factorize_ATAI(D, self.rho)

        # ADMM iteration
        X = torch.zeros((ND, N, K), dtype=torch.float, device=D.device)
        Y = torch.zeros((ND, N, K), dtype=torch.float, device=D.device)
        U = torch.zeros((ND, N, K), dtype=torch.float, device=D.device)

        Y_deep = []
        for stage in range(0, self.stage_number):
            X, Y, U = self.admm_iteration[stage](DTS, DTDIc, self.rho, self.lambdaV, X, Y, U)
            
            if stage in deep_supervision_stages:
                Y_deep.append(Y)

        self.Y = Y
        if return_sr_ds:
            self.Y_deep = Y_deep
            return Y, Y_deep
        else:
            return Y

    def freeze_dictionary(self, freeze=True):
        self.dictionary.requires_grad = not freeze

    def get_dictionary(self):
        return self.dictionary

    def set_dictionary(self, dictionary, train_dict=False):
        d0 = torch.tensor(dictionary, dtype=self.dictionary.dtype, device=self.dictionary.device)
        self._set_dictionary(d0, train_dict)

    def _set_dictionary(self, d0, train_dict=True):        
        d0 = F.normalize(d0, p=2, dim=-2)

        if self.dict_constraint != 1:
            d0 = d0 * self.dict_constraint

        #this won't work since d0 is computed from another variables
        #d0.requires_grad=train_dict 
        self.dictionary = nn.Parameter(d0, requires_grad=train_dict)

    def get_coefficient(self):
        return self.Y

    @staticmethod
    def _cholesky_factorize_ATAI(A, rho, upper=False) -> torch.Tensor:
        """
        Compute Cholesky factorization of $A^T A + \\rho I$
        This is to calculate the inverse of the matrix faster during update  

        TODO: 
        Let 
        M, N be size of A,        
        if M < N, it is more efficient to compute $AA^T + \\rho I$

        Args:
            A (Tensor): B x M x N
            rho (float):
            upper (bool): if create upper factorization 

        Returns:
            Tensor: Cholesky factorization of $A^T A + \\rho I$ 
        """
        ATA = torch.bmm(A.transpose(-1, -2), A)
        I = torch.eye(ATA.shape[-2], ATA.shape[-1], dtype=ATA.dtype, device=ATA.device)
        return torch.cholesky(ATA + rho * I, upper)


class AdmmBlock(nn.Module):
    """
    ADMM update iteration to solve sparse representation problem
    $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$

    """

    def __init__(self) -> None:
        super(AdmmBlock, self).__init__()

        # create update block
        self.x_update = XUpdateBlock()
        self.y_update = YUpdateBlock()
        self.u_update = UUpdateBlock()
        
    def forward(self, DTS, DTDIc, rho, lambdaV, X, Y, U):
        """
        ADMM update iteration to solve sparse representation problem
        $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$

        `DTS`: $D^T * S`

        `DTDIc`: Cholesky factorisation of $D^T * D + \\rho * I$
        """

        x = self.x_update(DTS, DTDIc, rho, Y, U)
        y = self.y_update(rho, lambdaV, x, U)
        u = self.u_update(x, y, U)

        return (x, y, u)


class XUpdateBlock(nn.Module):
    """
    ADMM x update step of sparse reconstruction problem

    $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$

    $x^{k+1}:=  (D^TD + \\rho I)^{−1} * (D^TS + \\rho y^k − z^k)$

    """

    def __init__(self) -> None:
        super(XUpdateBlock, self).__init__()
                    

    def forward(self, DTS, DTDIc, rho, Y, U) -> torch.Tensor:
        """
        ADMM x update step of sparse reconstruction problem

        $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$

        $x^{k+1}:=  (D^TD + \\rho I)^{−1} * (D^TS + \\rho y^k − z^k)$

        `DTS`: $D^T * S$

        `DTDIc`: Cholesky factorisation of $D^T * D + \\rho * I$

        """

        b = DTS + rho * (Y - U)        
        return self._cholesky_solve_ATAI(b, DTDIc)


    @staticmethod
    def _cholesky_solve_ATAI(b, c, upper=False) -> torch.Tensor:
        """
        Solve linear system $(A^T A + \\rho I)X = B$ using Cholesky method  
        given Cholesky factor matrix c
        """
        return torch.cholesky_solve(b, c, upper)    


class YUpdateBlock(nn.Module):
    """
    ADMM y update step of sparse reconstruction problem

    $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$

    $Y^{k+1} = ST(X^{k+1} + U^k, \lambda/\\rho)$
    """

    def __init__(self) -> None:
        super(YUpdateBlock, self).__init__()
                

    def forward(self, rho, lambdaV, X, U) -> torch.Tensor:
        """
        ADMM y update step of sparse reconstruction problem

        $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$

        $Y^{k+1} = ST(X^{k+1} + U^k, \lambda/\\rho)$
        where $ST(.)$ is soft thresholding

        `X`: ADMM primal variable

        `U`: ADMM scaled multiplier

        """

        return self._proximal_l1(X + U, (lambdaV / rho))
        

    @staticmethod
    def _proximal_l1(v, alpha) -> torch.Tensor:
        """
        Proximal operator of L1 norm, aka soft thresholding

        $S(v, a) = (v−a)_+−(−v−a)_+$
        """

        return torch.sign(v) * (torch.clamp(torch.abs(v) - alpha, min=0))


class UUpdateBlock(nn.Module):
    """
    ADMM u update step of sparse reconstruction problem

    $argmin_X (1/2)|DX-S|_2^2 + \lambda|X|_1$
    
    $u^{k+1} = u^k + (x^{k+1} − y^{k+1})$
    """

    def __init__(self) -> None:
        super(UUpdateBlock, self).__init__()      
        

    def forward(self, X, Y, U) -> torch.Tensor:
        return U + (X - Y)
