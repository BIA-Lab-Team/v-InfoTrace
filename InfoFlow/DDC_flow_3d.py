import numpy as np
from InfoFlow.DDC_flow import differential_covariance


def DDC_cause_3d(img1, img2, eps=1e-12, alpha=1e-2):
    """
    img1, img2: (wz, wxy, wxy, T) — anisotropic spatial patch + time
    Returns: (wz, wxy, wxy) causal score array
    """
    spatial_shape = img1.shape[:-1]
    T = img1.shape[-1]
    Y_ = np.array([img1, img2]).reshape(2, -1, T)
    W_ = differential_covariance(Y_.reshape(-1, T), alpha=alpha)
    N = Y_.shape[1]
    W_out = W_[:N, N:]
    return np.diag(W_out).reshape(spatial_shape)
