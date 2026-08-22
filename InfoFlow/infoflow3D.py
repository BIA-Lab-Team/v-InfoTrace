#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D volumetric extension of InfoFlow/infoflow.py.

Input volumes have shape (T, Z, H, W). Outputs are (Z, H, W, 3) flow vectors
or (Z, H, W, wz, wxy, wxy) score arrays. Window sizes are anisotropic:
- winsize_xy : window half-size along y and x axes
- winsize_z  : window half-size along the z (depth) axis

Sequential and parallel (joblib) variants are provided for every orchestrator.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_field_shape_3d(vol_size, search_area_size, overlap):
    """Return (n_z, n_y, n_x) window counts given 3-element tuples."""
    vol_size = np.asarray(vol_size)
    search_area_size = np.asarray(search_area_size)
    overlap = np.asarray(overlap)
    return (vol_size - search_area_size) // (search_area_size - overlap) + 1


def get_coordinates_3d(vol_size, search_area_size, overlap):
    """Return (Zg, Yg, Xg) meshgrid of window-centre voxel indices."""
    field_shape = get_field_shape_3d(vol_size, search_area_size, overlap)
    stride = np.asarray(search_area_size) - np.asarray(overlap)
    z = np.arange(field_shape[0]) * stride[0] + search_area_size[0] / 2.0
    y = np.arange(field_shape[1]) * stride[1] + search_area_size[1] / 2.0
    x = np.arange(field_shape[2]) * stride[2] + search_area_size[2] / 2.0
    return np.meshgrid(z, y, x, indexing='ij')


def sliding_window_array_time_3d(volume, window_size, overlap):
    """
    Extract all sliding windows from a (Z, H, W, T) volume.

    Parameters
    ----------
    volume       : (Z, H, W, T) ndarray
    window_size  : [ws_z, ws_h, ws_w]
    overlap      : [ov_z, ov_h, ov_w]

    Returns
    -------
    windows : (N_windows, ws_z, ws_h, ws_w, T) ndarray
    """
    ws_z, ws_h, ws_w = window_size
    ov_z, ov_h, ov_w = overlap
    stride_z = ws_z - ov_z
    stride_h = ws_h - ov_h
    stride_w = ws_w - ov_w
    Z, H, W = volume.shape[:3]

    n_z = (Z - ws_z) // stride_z + 1
    n_h = (H - ws_h) // stride_h + 1
    n_w = (W - ws_w) // stride_w + 1

    z0 = (np.arange(n_z) * stride_z).astype(int)
    y0 = (np.arange(n_h) * stride_h).astype(int)
    x0 = (np.arange(n_w) * stride_w).astype(int)

    Z0, Y0, X0 = np.meshgrid(z0, y0, x0, indexing='ij')   # (n_z, n_h, n_w)
    Z0 = Z0.reshape(-1, 1, 1, 1)
    Y0 = Y0.reshape(-1, 1, 1, 1)
    X0 = X0.reshape(-1, 1, 1, 1)

    iz, iy, ix = np.meshgrid(np.arange(ws_z), np.arange(ws_h), np.arange(ws_w), indexing='ij')
    win_z = Z0 + iz   # (N, ws_z, ws_h, ws_w)
    win_y = Y0 + iy
    win_x = X0 + ix

    return volume[win_z, win_y, win_x]   # (N, ws_z, ws_h, ws_w, T)


# ---------------------------------------------------------------------------
# Vectorised gradient (score → 3-component flow)
# ---------------------------------------------------------------------------

def causal_block_flow_scores_gradient_3d(scores):
    """
    Convert a 6-D score array to 3-component flow vectors.

    Parameters
    ----------
    scores : (nZ, nH, nW, wz, wxy, wxy) ndarray

    Returns
    -------
    flow : (nZ, nH, nW, 3) ndarray  — (z, y, x) components

    Notes
    -----
    Slicing uses symmetric halves: lower = scores[..., :mid, ...],
    upper = scores[..., sz-mid:, ...].  For odd sz the centre element
    is excluded; for even sz both halves have the same number of elements.
    This avoids the empty-slice artifact that occurs with mid+1: on an
    axis of size 2 (e.g. winsize_z=2).
    """
    sz  = scores.shape[-3]
    sxy = scores.shape[-1]
    mid_z  = sz  // 2
    mid_xy = sxy // 2

    cz = (-np.apply_over_axes(np.nansum, scores[..., :mid_z,       :,          :          ], [-3, -2, -1])
          + np.apply_over_axes(np.nansum, scores[..., sz-mid_z:,    :,          :          ], [-3, -2, -1]))
    cy = (-np.apply_over_axes(np.nansum, scores[..., :,             :mid_xy,    :          ], [-3, -2, -1])
          + np.apply_over_axes(np.nansum, scores[..., :,             sxy-mid_xy:,:          ], [-3, -2, -1]))
    cx = (-np.apply_over_axes(np.nansum, scores[..., :,             :,          :mid_xy    ], [-3, -2, -1])
          + np.apply_over_axes(np.nansum, scores[..., :,             :,          sxy-mid_xy:], [-3, -2, -1]))

    intensity = np.nansum(scores.reshape(*scores.shape[:3], -1), axis=-1)
    flow = -np.stack([np.squeeze(cz, axis=(-3, -2, -1)),
                      np.squeeze(cy, axis=(-3, -2, -1)),
                      np.squeeze(cx, axis=(-3, -2, -1))], axis=-1) * intensity[..., None]
    return flow


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for joblib loky pickling)
# ---------------------------------------------------------------------------

def _worker_causal_flow_3d(args):
    window, wz, wxy, cause_fnc, kwargs = args
    patch = window[wz:2*wz, wxy:2*wxy, wxy:2*wxy]
    corr_array = cause_fnc(patch, patch, **kwargs).reshape((wz, wxy, wxy))
    sz, sxy = corr_array.shape[0], corr_array.shape[1]
    mid_z, mid_xy = sz // 2, sxy // 2
    cz = (-np.nansum(corr_array[:mid_z,      :,          :         ])
          + np.nansum(corr_array[sz-mid_z:,   :,          :         ]))
    cy = (-np.nansum(corr_array[:,            :mid_xy,    :         ])
          + np.nansum(corr_array[:,            sxy-mid_xy:,:         ]))
    cx = (-np.nansum(corr_array[:,            :,          :mid_xy   ])
          + np.nansum(corr_array[:,            :,          sxy-mid_xy:]))
    intensity = np.nansum(corr_array)
    return -(np.array([cz, cy, cx]) * intensity)


def _worker_causal_flow_scores_3d(args):
    window, wz, wxy, cause_fnc, kwargs = args
    patch = window[wz:2*wz, wxy:2*wxy, wxy:2*wxy]
    return cause_fnc(patch, patch, **kwargs).reshape((wz, wxy, wxy))


def _worker_causal_block_flow_3d(args):
    window, wz, wxy, cause_fnc, kwargs = args
    corr_array = cause_fnc(window, window, **kwargs).reshape((wz, wxy, wxy))
    sz, sxy = corr_array.shape[0], corr_array.shape[1]
    mid_z, mid_xy = sz // 2, sxy // 2
    cz = (-np.nansum(corr_array[:mid_z,      :,          :         ])
          + np.nansum(corr_array[sz-mid_z:,   :,          :         ]))
    cy = (-np.nansum(corr_array[:,            :mid_xy,    :         ])
          + np.nansum(corr_array[:,            sxy-mid_xy:,:         ]))
    cx = (-np.nansum(corr_array[:,            :,          :mid_xy   ])
          + np.nansum(corr_array[:,            :,          sxy-mid_xy:]))
    intensity = np.nansum(corr_array)
    return np.array([cz, cy, cx]) * intensity


# ---------------------------------------------------------------------------
# Sequential orchestrators
# ---------------------------------------------------------------------------

def causal_flow_3d(vol, cause_fnc, winsize_xy=3, winsize_z=3, **kwargs):
    """
    Compute per-voxel 3-component causal flow vectors (sequential).

    Parameters
    ----------
    vol       : (T, Z, H, W) ndarray
    cause_fnc : callable(img1, img2, **kw) → 1-D array of length wz*wxy*wxy
    winsize_xy, winsize_z : window half-sizes

    Returns
    -------
    flow : (Z, H, W, 3) ndarray — (z, y, x) components
    """
    import scipy.ndimage as ndimage
    import skimage.transform as sktform
    from tqdm import tqdm

    wz, wxy = winsize_z, winsize_xy
    frame_ = np.pad(vol.transpose(1, 2, 3, 0),
                    [[wz, wz], [wxy, wxy], [wxy, wxy], [0, 0]],
                    mode='constant', constant_values=0)
    windows_3 = sliding_window_array_time_3d(
        frame_,
        window_size=[3*wz, 3*wxy, 3*wxy],
        overlap=[2*wz, 2*wxy, 2*wxy])
    Zg, Yg, Xg = get_coordinates_3d(frame_.shape[:3],
                                     [3*wz, 3*wxy, 3*wxy],
                                     [2*wz, 2*wxy, 2*wxy])
    grid_shape = Zg.shape

    GC_vectors = []
    for ii in tqdm(range(len(windows_3))):
        patch = windows_3[ii, wz:2*wz, wxy:2*wxy, wxy:2*wxy]
        corr_array = cause_fnc(patch, patch, **kwargs).reshape((wz, wxy, wxy))
        sz, sxy = corr_array.shape[0], corr_array.shape[1]
        mid_z, mid_xy = sz // 2, sxy // 2
        cz = (-np.nansum(corr_array[:mid_z,      :,          :         ])
              + np.nansum(corr_array[sz-mid_z:,   :,          :         ]))
        cy = (-np.nansum(corr_array[:,            :mid_xy,    :         ])
              + np.nansum(corr_array[:,            sxy-mid_xy:,:         ]))
        cx = (-np.nansum(corr_array[:,            :,          :mid_xy   ])
              + np.nansum(corr_array[:,            :,          sxy-mid_xy:]))
        intensity = np.nansum(corr_array)
        GC_vectors.append(-(np.array([cz, cy, cx]) * intensity))

    GC_vectors = np.array(GC_vectors).reshape(grid_shape + (3,))
    for ch in range(3):
        GC_vectors[..., ch] = ndimage.gaussian_filter(GC_vectors[..., ch], sigma=1.)
    GC_vectors = np.stack([
        sktform.resize(GC_vectors[..., ch], output_shape=vol.shape[1:],
                       preserve_range=True, order=1)
        for ch in range(3)], axis=-1)
    return GC_vectors


def causal_flow_scores_3d(vol, cause_fnc, winsize_xy=3, winsize_z=3, **kwargs):
    """
    Accumulate per-voxel causal score blocks (sequential).

    Returns
    -------
    scores : (Z, H, W, wz, wxy, wxy) ndarray
    """
    import skimage.transform as sktform
    from tqdm import tqdm

    wz, wxy = winsize_z, winsize_xy
    frame_ = np.pad(vol.transpose(1, 2, 3, 0),
                    [[wz, wz], [wxy, wxy], [wxy, wxy], [0, 0]],
                    mode='constant', constant_values=0)
    windows_3 = sliding_window_array_time_3d(
        frame_,
        window_size=[3*wz, 3*wxy, 3*wxy],
        overlap=[2*wz, 2*wxy, 2*wxy])
    Zg, Yg, Xg = get_coordinates_3d(frame_.shape[:3],
                                     [3*wz, 3*wxy, 3*wxy],
                                     [2*wz, 2*wxy, 2*wxy])
    grid_shape = Zg.shape

    GC_scores = []
    for ii in tqdm(range(len(windows_3))):
        patch = windows_3[ii, wz:2*wz, wxy:2*wxy, wxy:2*wxy]
        corr_array = cause_fnc(patch, patch, **kwargs).reshape((wz, wxy, wxy))
        GC_scores.append(corr_array)

    GC_scores = np.array(GC_scores).reshape(grid_shape + (wz, wxy, wxy))
    GC_scores = sktform.resize(GC_scores,
                               output_shape=vol.shape[1:] + (wz, wxy, wxy),
                               preserve_range=True, order=1)
    return GC_scores


def causal_block_flow_3d(vol, cause_fnc, winsize_xy=3, winsize_z=3, **kwargs):
    """
    Compute per-voxel 3-component causal flow using full window blocks (sequential).
    For use with block-structured measures like PCCA.

    Returns
    -------
    flow : (Z, H, W, 3) ndarray
    """
    import scipy.ndimage as ndimage
    import skimage.transform as sktform
    from tqdm import tqdm

    wz, wxy = winsize_z, winsize_xy
    frame_ = np.pad(vol.transpose(1, 2, 3, 0),
                    [[wz, wz], [wxy, wxy], [wxy, wxy], [0, 0]],
                    mode='constant', constant_values=0)
    windows_3 = sliding_window_array_time_3d(
        frame_,
        window_size=[3*wz, 3*wxy, 3*wxy],
        overlap=[2*wz, 2*wxy, 2*wxy])
    Zg, Yg, Xg = get_coordinates_3d(frame_.shape[:3],
                                     [3*wz, 3*wxy, 3*wxy],
                                     [2*wz, 2*wxy, 2*wxy])
    grid_shape = Zg.shape

    GC_vectors = []
    for ii in tqdm(range(len(windows_3))):
        corr_array = cause_fnc(windows_3[ii], windows_3[ii], **kwargs).reshape((wz, wxy, wxy))
        sz, sxy = corr_array.shape[0], corr_array.shape[1]
        mid_z, mid_xy = sz // 2, sxy // 2
        cz = (-np.nansum(corr_array[:mid_z,      :,          :         ])
              + np.nansum(corr_array[sz-mid_z:,   :,          :         ]))
        cy = (-np.nansum(corr_array[:,            :mid_xy,    :         ])
              + np.nansum(corr_array[:,            sxy-mid_xy:,:         ]))
        cx = (-np.nansum(corr_array[:,            :,          :mid_xy   ])
              + np.nansum(corr_array[:,            :,          sxy-mid_xy:]))
        intensity = np.nansum(corr_array)
        GC_vectors.append(np.array([cz, cy, cx]) * intensity)

    GC_vectors = np.array(GC_vectors).reshape(grid_shape + (3,))
    for ch in range(3):
        GC_vectors[..., ch] = ndimage.gaussian_filter(GC_vectors[..., ch], sigma=1.)
    GC_vectors = np.stack([
        sktform.resize(GC_vectors[..., ch], output_shape=vol.shape[1:],
                       preserve_range=True, order=1)
        for ch in range(3)], axis=-1)
    return GC_vectors


# ---------------------------------------------------------------------------
# Parallel orchestrators
# ---------------------------------------------------------------------------

def causal_flow_parallel_3d(vol, cause_fnc, winsize_xy=3, winsize_z=3, n_jobs=-1, **kwargs):
    """Parallel version of causal_flow_3d."""
    from joblib import Parallel, delayed
    import scipy.ndimage as ndimage
    import skimage.transform as sktform
    from tqdm import tqdm

    wz, wxy = winsize_z, winsize_xy
    frame_ = np.pad(vol.transpose(1, 2, 3, 0),
                    [[wz, wz], [wxy, wxy], [wxy, wxy], [0, 0]],
                    mode='constant', constant_values=0)
    windows_3 = sliding_window_array_time_3d(
        frame_,
        window_size=[3*wz, 3*wxy, 3*wxy],
        overlap=[2*wz, 2*wxy, 2*wxy])
    Zg, Yg, Xg = get_coordinates_3d(frame_.shape[:3],
                                     [3*wz, 3*wxy, 3*wxy],
                                     [2*wz, 2*wxy, 2*wxy])
    grid_shape = Zg.shape

    args_list = [(windows_3[ii], wz, wxy, cause_fnc, kwargs) for ii in range(len(windows_3))]
    GC_vectors = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_flow_3d)(a) for a in tqdm(args_list))

    GC_vectors = np.array(GC_vectors).reshape(grid_shape + (3,))
    for ch in range(3):
        GC_vectors[..., ch] = ndimage.gaussian_filter(GC_vectors[..., ch], sigma=1.)
    GC_vectors = np.stack([
        sktform.resize(GC_vectors[..., ch], output_shape=vol.shape[1:],
                       preserve_range=True, order=1)
        for ch in range(3)], axis=-1)
    return GC_vectors


def causal_flow_scores_parallel_3d(vol, cause_fnc, winsize_xy=3, winsize_z=3, n_jobs=-1, **kwargs):
    """Parallel version of causal_flow_scores_3d."""
    from joblib import Parallel, delayed
    import skimage.transform as sktform
    from tqdm import tqdm

    wz, wxy = winsize_z, winsize_xy
    frame_ = np.pad(vol.transpose(1, 2, 3, 0),
                    [[wz, wz], [wxy, wxy], [wxy, wxy], [0, 0]],
                    mode='constant', constant_values=0)
    windows_3 = sliding_window_array_time_3d(
        frame_,
        window_size=[3*wz, 3*wxy, 3*wxy],
        overlap=[2*wz, 2*wxy, 2*wxy])
    Zg, Yg, Xg = get_coordinates_3d(frame_.shape[:3],
                                     [3*wz, 3*wxy, 3*wxy],
                                     [2*wz, 2*wxy, 2*wxy])
    grid_shape = Zg.shape

    args_list = [(windows_3[ii], wz, wxy, cause_fnc, kwargs) for ii in range(len(windows_3))]
    GC_scores = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_flow_scores_3d)(a) for a in tqdm(args_list))

    GC_scores = np.array(GC_scores).reshape(grid_shape + (wz, wxy, wxy))
    GC_scores = sktform.resize(GC_scores,
                               output_shape=vol.shape[1:] + (wz, wxy, wxy),
                               preserve_range=True, order=1)
    return GC_scores


def causal_block_flow_parallel_3d(vol, cause_fnc, winsize_xy=3, winsize_z=3, n_jobs=-1, **kwargs):
    """Parallel version of causal_block_flow_3d."""
    from joblib import Parallel, delayed
    import scipy.ndimage as ndimage
    import skimage.transform as sktform
    from tqdm import tqdm

    wz, wxy = winsize_z, winsize_xy
    frame_ = np.pad(vol.transpose(1, 2, 3, 0),
                    [[wz, wz], [wxy, wxy], [wxy, wxy], [0, 0]],
                    mode='constant', constant_values=0)
    windows_3 = sliding_window_array_time_3d(
        frame_,
        window_size=[3*wz, 3*wxy, 3*wxy],
        overlap=[2*wz, 2*wxy, 2*wxy])
    Zg, Yg, Xg = get_coordinates_3d(frame_.shape[:3],
                                     [3*wz, 3*wxy, 3*wxy],
                                     [2*wz, 2*wxy, 2*wxy])
    grid_shape = Zg.shape

    args_list = [(windows_3[ii], wz, wxy, cause_fnc, kwargs) for ii in range(len(windows_3))]
    GC_vectors = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_block_flow_3d)(a) for a in tqdm(args_list))

    GC_vectors = np.array(GC_vectors).reshape(grid_shape + (3,))
    for ch in range(3):
        GC_vectors[..., ch] = ndimage.gaussian_filter(GC_vectors[..., ch], sigma=1.)
    GC_vectors = np.stack([
        sktform.resize(GC_vectors[..., ch], output_shape=vol.shape[1:],
                       preserve_range=True, order=1)
        for ch in range(3)], axis=-1)
    return GC_vectors
