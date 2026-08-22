import numpy as np
from InfoFlow.pdc_dtf_flow import mvar_fit, PDC


def PDC_central_flow_3d(img1, img2, lag=5, alpha=.1, random_state=0):
    """
    img1, img2: (wz, wxy, wxy, T) — anisotropic spatial patch + time
    Returns: (wz, wxy, wxy) causal score array
    """
    spatial_shape = img1.shape[:-1]
    Y_ = np.array([img1, img2])
    Y_ = Y_.reshape(len(Y_), -1, Y_.shape[-1]).reshape(-1, Y_.shape[-1])
    A_est, sigma = mvar_fit(Y_, p=lag)
    sigma = np.abs(np.diag(sigma))
    N = int(Y_.shape[0] // 2)
    P, freqs = PDC(A_est, sigma)
    P_sum = np.nansum(P, axis=0)
    P_block = P_sum[N:, :N].copy()
    corr_array = np.zeros(spatial_shape)
    for ind in range(N):
        idx = np.unravel_index(ind, spatial_shape)
        corr_array[idx] = P_block[ind, ind]
    return corr_array
