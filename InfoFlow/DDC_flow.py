#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:15 2022

@author: s434626
"""

def differential_covariance(X, eps=1e-12, alpha=1e-3):
    """ uses ridge regularization 
    """
    import numpy as np 
    
    # standardize
    # X_ = (X - np.nanmean(X, axis=1)[:,None])  / ( np.nanstd(X, axis=1)[:,None] + eps )
    X_ = X.copy()
    
    # # differential 
    # X_pad = np.pad(X_, pad_width=[[0,0],[1,1]], mode='edge')
    # dX_ = (X_pad[:,2:] - X_pad[:,:-2]) / 2.
    # # dX_ = np.gradient(X_, axis=1)
    dX_ = X_[:,1:] - X_[:,:-1]
    X_ = X_[:,1:].copy()
    # print(X_.shape,dX_.shape)
    X_ = X_.T
    dX_ = dX_.T
    
    # linear least squares solution . 
    dX_X = dX_.T.dot(X_)
    X_X = X_.T.dot(X_)
    
    # W = np.linalg.solve(X_X+reg*np.eye(len(X_X)), dX_X) # transpose... 
    W = dX_X.dot(np.linalg.inv(X_X +alpha*np.eye(len(X_X))))
    
    return W 


def DDC_cause(img1, img2, eps=1e-12, alpha=1e-2):

    import numpy as np 
    import pylab as plt 
    
    # compile all the timeseries
    Y_ = np.array([img1, 
                   img2])
    Y_ = Y_.reshape(len(Y_), -1, Y_.shape[-1])

    """
    Compute the diff covariance 
    """
    W_ = differential_covariance(Y_.reshape(-1, Y_.shape[-1]), eps=eps, alpha=alpha)
    
    # W_ = W_.T.copy()
    N = Y_.shape[1] # this is the flattened over spatial windows. 
    N_rows = int(np.sqrt(N))

    # W_out = W_[1:,0].copy() # - W_[0,1:]
    W_out = W_[:N, N:].copy()
    corr_array = np.zeros((N_rows,N_rows))

    for ii in np.arange(N_rows):
        for jj in np.arange(N_rows):
            ind = ii*N_rows + jj
            corr_array[ii,jj] = W_out[ind,ind]

    return corr_array


def _differential_covariance_lagged(target_flat, confound_flats, candidate_flat, lag, eps=1e-12, alpha=1e-2):
    """
    Explicit-design-matrix lagged generalization of differential_covariance.

    differential_covariance itself is a zero-lag/contemporaneous Jacobian
    estimator: it relates each stacked variable's *current* value to
    everyone's *simultaneous* derivative. That cannot see a genuinely lagged
    relationship -- e.g. a moving pattern that takes >=1 frame to travel from
    one spatial block to a non-overlapping neighbor -- and naively
    time-shifting an input before handing it to differential_covariance
    doesn't fix this, since differential_covariance re-differences internally
    and the two shifts interact to cancel back out to near zero-lag.

    Instead this builds the regression directly: target's own derivative
    (dX/dt at time t, t+1) is regressed on target's own contemporaneous state
    (the autoregressive baseline) and every confound's contemporaneous state
    (both at t+1, matching differential_covariance's own convention), plus
    candidate's state `lag` steps EARLIER (at t+1-lag) -- a genuinely delayed
    predictor, distinct from the zero-lag baseline/confound terms.

    target_flat, candidate_flat : (n, T) arrays
    confound_flats : list of (n_i, T) arrays
    """
    import numpy as np

    T = target_flat.shape[1]
    idx = np.arange(lag, T - 1)  # derivative window start times; need candidate at idx-lag >= 0

    dX = target_flat[:, idx + 1] - target_flat[:, idx]
    X_self = target_flat[:, idx + 1]
    X_conf = [c[:, idx + 1] for c in confound_flats]
    X_cand = candidate_flat[:, idx - lag]

    X = np.vstack([X_self] + X_conf + [X_cand])

    dX_X = dX.dot(X.T)
    X_X = X.dot(X.T)
    W = dX_X.dot(np.linalg.inv(X_X + alpha * np.eye(X_X.shape[0])))

    return W


def DDC_cause_confound(target, confounds, candidate, eps=1e-12, alpha=1e-2, lag=1):
    """
    Confound-aware, lag-aware variant of DDC_cause -- see
    _differential_covariance_lagged for why an explicit lagged design is
    needed rather than reusing differential_covariance directly.

    target, candidate : (*, T) arrays
    confounds : list of (*, T) arrays
    """
    import numpy as np

    def _flat(arr):
        return arr.reshape(-1, arr.shape[-1])  # (n, T)

    target_flat = _flat(target)
    confound_flats = [_flat(c) for c in confounds]
    candidate_flat = _flat(candidate)

    W_ = _differential_covariance_lagged(target_flat, confound_flats, candidate_flat, lag, eps=eps, alpha=alpha)

    N_c = candidate_flat.shape[0]
    W_out = W_[:, -N_c:]

    return np.nanmean(W_out)

