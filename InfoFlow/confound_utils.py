#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for confound-aware (partial) causal measures used by the
multi-channel information-flow extension: residualizing a signal against
one or more confound blocks via ridge regression, the standard technique
behind partial correlation / partial canonical correlation.
"""

import numpy as np


def residualize_against_confounds(signal, confounds, alpha=1.0):
    """
    Regress `signal` on the confound blocks and return the residual,
    pixel-by-pixel over the flattened spatial dimension of `signal`.

    Each confound block is summarized by its spatial-mean time series (one
    regressor column per confound), rather than one column per confound
    pixel: a block can have many more pixels than there are timepoints T, in
    which case a per-pixel design matrix is underdetermined and even mild
    ridge regularization overfits it, washing out genuine signal along with
    the confound. Using the aggregate signal keeps the regression
    well-conditioned while still controlling for each confound's dominant
    contemporaneous drive.

    signal : array, shape (..., T)
    confounds : list of arrays, each shape (..., T) (any spatial shape, same T)
    alpha : ridge penalty, scaled by the confound design matrix's average
        variance so the effective regularization strength is scale-invariant.
    """
    if len(confounds) == 0:
        return signal

    T = signal.shape[-1]
    sig_flat = signal.reshape(-1, T).T  # (T, n_sig)

    conf_flat = np.column_stack([c.reshape(-1, T).mean(axis=0) for c in confounds])  # (T, n_conf)
    conf_flat = conf_flat - conf_flat.mean(axis=0, keepdims=True)

    sig_mean = sig_flat.mean(axis=0, keepdims=True)
    sig_centered = sig_flat - sig_mean

    XtX = conf_flat.T.dot(conf_flat)
    alpha_eff = alpha * (np.trace(XtX) / XtX.shape[0] + 1e-12)
    beta = np.linalg.solve(XtX + alpha_eff * np.eye(XtX.shape[0]), conf_flat.T.dot(sig_centered))
    residual = sig_centered - conf_flat.dot(beta)

    residual_full = residual + sig_mean
    return residual_full.T.reshape(signal.shape)
