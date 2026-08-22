import numpy as np
from InfoFlow.LK_flow import Linear_LK


def Linear_LK_cause_3d(img1, img2, eps=1e-12, alpha=1e-2):
    """
    img1, img2: (wz, wxy, wxy, T) — anisotropic spatial patch + time
    Returns: (wz, wxy, wxy) causal score array
    """
    spatial_shape = img1.shape[:-1]
    Y_ = np.array([img1, img2])
    Y_ = Y_.reshape(len(Y_), -1, Y_.shape[-1])
    W_, C_ = Linear_LK(Y_.reshape(-1, Y_.shape[-1]), eps=eps, alpha=alpha)
    N = Y_.shape[1]
    W_out = W_[:N, N:].copy()
    corr_array = np.zeros(spatial_shape)
    for ind in range(N):
        idx = np.unravel_index(ind, spatial_shape)
        corr_array[idx] = W_out[ind, ind] * np.sqrt(C_[ind, ind + N] / C_[ind, ind] + eps)
    return corr_array
