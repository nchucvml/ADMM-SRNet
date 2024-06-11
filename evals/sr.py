import torch

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