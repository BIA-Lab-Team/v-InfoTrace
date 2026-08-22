#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:15 2022

@author: s205272
"""


import numpy as np


def _snap_odd_winsize(winsize):
    """
    winsize must be odd so every sub-block has a well-defined single center
    pixel. Rather than rejecting even input, snap to the nearest odd integer
    and warn.
    """
    import warnings
    snapped = (winsize // 2) * 2 + 1
    if snapped != winsize:
        warnings.warn(f"winsize={winsize} is even; snapping to nearest odd value {snapped}.")
    return snapped


def _prepare_windows(vid, winsize):
    """
    Shared pad + sliding-window + coordinate-grid setup used by all the
    causal_flow*/causal_block_flow* drivers.
    """
    frame_ = np.pad(vid.transpose(1,2,0), [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
    windows_3 = sliding_window_array_time(frame_, window_size=(3*winsize), overlap=2*winsize)
    x, y = get_coordinates(image_size=frame_[:,:,0].shape,
                            search_area_size=(3*winsize), overlap=2*winsize)
    xy_coords = np.dstack([x,y])
    return windows_3, xy_coords


def _worker_causal_flow(args):
    window, winsize, cause_fnc, kwargs = args
    corr_array = cause_fnc(
        window[winsize:2*winsize, winsize:2*winsize],
        window[winsize:2*winsize, winsize:2*winsize],
        **kwargs
    ).reshape((winsize, winsize))
    mid = corr_array.shape[1] // 2
    cx = -np.nansum(corr_array[:, :mid]) + np.nansum(corr_array[:, mid+1:])
    cy = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
    intensity = np.nansum(corr_array)
    return -(np.hstack([cy, cx]) * intensity)


def _worker_causal_flow_scores(args):
    window, winsize, cause_fnc, kwargs = args
    return cause_fnc(
        window[winsize:2*winsize, winsize:2*winsize],
        window[winsize:2*winsize, winsize:2*winsize],
        **kwargs
    ).reshape((winsize, winsize))


def _worker_causal_block_flow(args):
    window, winsize, cause_fnc, kwargs = args
    corr_array = cause_fnc(window, window, **kwargs).reshape((winsize, winsize))
    mid = corr_array.shape[1] // 2
    cx = -np.nansum(corr_array[:, :mid]) + np.nansum(corr_array[:, mid+1:])
    cy = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
    intensity = np.nansum(corr_array)
    return np.hstack([cy, cx]) * intensity


def get_field_shape(image_size, search_area_size, overlap):
    """Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape
    search_area_size: tuple
        the size of the interrogation windows (if equal in frames A,B)
        or the search area (in frame B), the largest  of the two
    overlap: tuple
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    field_shape : 2-element tuple
        the shape of the resulting flow field
    """
    import numpy as np 
    
    field_shape = (np.array(image_size) - np.array(search_area_size)) // (
        np.array(search_area_size) - np.array(overlap)
    ) + 1
    
    return field_shape


def get_coordinates(image_size, search_area_size, overlap, center_on_field = True):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    search_area_size: int
        the size of the search area windows, sometimes it's equal to
        the interrogation window size in both frames A and B
    overlap: int = 0 (default is no overlap)
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
        Coordinate system 0,0 is at the top left corner, positive
        x to the right, positive y from top downwards, i.e.
        image coordinate system
    """
    import numpy as np 
    
    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)
    # print(len(field_shape))
    # print(field_shape)
    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = (
        np.arange(field_shape[1]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[0]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )

    # moving coordinates further to the center, so that the points at the
    # extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer
    # movements are allowed.
    if center_on_field == True:
        x += (
            image_size[1]
            - 1
            - ((field_shape[1] - 1) * (search_area_size - overlap) +
                (search_area_size - 1))
        ) // 2
        y += (
            image_size[0] - 1
            - ((field_shape[0] - 1) * (search_area_size - overlap) +
               (search_area_size - 1))
        ) // 2

        # the origin 0,0 is at top left
        # the units are pixels

    return np.meshgrid(x, y)

def get_rect_coordinates(frame_a, window_size, overlap, center_on_field = False):
    '''
    Rectangular grid version of get_coordinates.
    '''
    import numpy as np 
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]
        
    _, y = get_coordinates(frame_a, window_size[0], overlap[0], center_on_field = False)
    x, _ = get_coordinates(frame_a, window_size[1], overlap[1], center_on_field = False)
    
    return np.meshgrid(x[0,:], y[:,0])



def sliding_window_array(image, window_size = 64, overlap = 32):
    '''
    This version does not use numpy as_strided and is much more memory efficient.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window. 
    '''
    import numpy as np 
    
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    x, y = get_rect_coordinates(image.shape, window_size, overlap, center_on_field = False)
    x = (x - window_size[1]//2).astype(int); y = (y - window_size[0]//2).astype(int)
    x, y = np.reshape(x, (-1,1,1)), np.reshape(y, (-1,1,1))

    win_x, win_y = np.meshgrid(np.arange(0, window_size[1]), np.arange(0, window_size[0]))
    win_x = win_x[np.newaxis,:,:] + x
    win_y = win_y[np.newaxis,:,:] + y
    
    # print(win_x.shape, win_y.shape)
    windows = image[win_y, win_x]
    
    return windows


def sliding_window_array_time(image, window_size = 64, overlap = 32):
    '''
    This version does not use numpy as_strided and is much more memory efficient.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window. 
    '''
    import numpy as np 
    
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    x, y = get_rect_coordinates(image.shape[:-1], window_size, overlap, center_on_field = False)
    x = (x - window_size[1]//2).astype(int); y = (y - window_size[0]//2).astype(int)
    x, y = np.reshape(x, (-1,1,1)), np.reshape(y, (-1,1,1))

    win_x, win_y = np.meshgrid(np.arange(0, window_size[1]), np.arange(0, window_size[0]))
    win_x = win_x[np.newaxis,:,:] + x
    win_y = win_y[np.newaxis,:,:] + y
    
    # print(win_x.shape, win_y.shape)
    windows = image[win_y, win_x]
    
    return windows


def moving_window_array(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.
    """
    import numpy as np 
    
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (
        sz * shape[1] * (window_size - overlap),
        sz * (window_size - overlap),
        sz * shape[1],
        sz,
    )
    shape = (
        int((shape[0] - window_size) / (window_size - overlap)) + 1,
        int((shape[1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )

    return np.lib.stride_tricks.as_strided(
        array, strides=strides, shape=shape
    ).reshape(-1, window_size, window_size)


def moving_window_array_time(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.
    """
    import numpy as np 
    
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (
        sz * shape[1] * (window_size - overlap),
        sz * (window_size - overlap),
        sz * shape[1],
        sz,
    )
    shape = (
        int((shape[0] - window_size) / (window_size - overlap)) + 1,
        int((shape[1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )

    return np.lib.stride_tricks.as_strided(
        array, strides=strides, shape=shape
    ).reshape(-1, window_size, window_size)


def read_video_cv2(avifile):
    
    import cv2
    
    vidcap = cv2.VideoCapture(avifile)
    success,image = vidcap.read()
    
    vid_array = []
    
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if success:
            vid_array.append(image)
        count += 1
        
    vid_array = np.array(vid_array)
      
    return vid_array


def causal_flow(vid, cause_fnc, winsize=3, **kwargs):

    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import numpy as np
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    windows_3, xy_coords = _prepare_windows(vid, winsize)

    GC_vectors = []
    
    # for ii in tqdm(np.arange(len(windows_3))):
    for ii in tqdm(np.arange(len(windows_3))):
        
        """
        This should allow usage of all functions that have the same call signature.
        """
        corr_array = cause_fnc(windows_3[ii, winsize:2*winsize, winsize:2*winsize], 
                               windows_3[ii, winsize:2*winsize, winsize:2*winsize], **kwargs) 
        # corr_array[1,1] = 0
        # corr_array = GC_full_reduced_separate_regress_individual(windows_3[ii], 
        #                                                           windows_3[ii], 
        #                                                               lag=lag, alpha=alpha) #(centre_ref, center)
        corr_array = corr_array.reshape((winsize,winsize))
        mid = corr_array.shape[1]//2
            
        corr_x_direction = -np.nansum(corr_array[:,:mid]) + np.nansum(corr_array[:,mid+1:])
        corr_y_direction = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
        intensity = np.nansum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
        
        mean_vector = np.hstack([corr_y_direction, corr_x_direction])
        mean_vector = mean_vector * intensity
        
        GC_vectors.append(-mean_vector)
        # GC_vectors.append(mean_vector)
            
    GC_vectors = np.array(GC_vectors).reshape((xy_coords.shape))
    GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    
    GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    
    return GC_vectors


def causal_flow_scores(vid, cause_fnc, winsize=3, **kwargs):

    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import numpy as np
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    windows_3, xy_coords = _prepare_windows(vid, winsize)

    GC_scores = []
    
    # for ii in tqdm(np.arange(len(windows_3))):
    for ii in tqdm(np.arange(len(windows_3))):
        
        """
        This should allow usage of all functions that have the same call signature.
        """
        corr_array = cause_fnc(windows_3[ii, winsize:2*winsize, winsize:2*winsize], 
                               windows_3[ii, winsize:2*winsize, winsize:2*winsize], **kwargs) 
        # corr_array[1,1] = 0
        # corr_array = GC_full_reduced_separate_regress_individual(windows_3[ii], 
        #                                                           windows_3[ii], 
        #                                                               lag=lag, alpha=alpha) #(centre_ref, center)
        corr_array = corr_array.reshape((winsize,winsize))
        
        GC_scores.append(corr_array)
        # mid = corr_array.shape[1]//2
            
        # corr_x_direction = -np.nansum(corr_array[:,:mid]) + np.nansum(corr_array[:,mid+1:])
        # corr_y_direction = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
        # intensity = np.nansum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
        
        # mean_vector = np.hstack([corr_y_direction, corr_x_direction])
        # mean_vector = mean_vector * intensity
        
        # GC_vectors.append(-mean_vector)
            
    GC_scores = np.array(GC_scores).reshape((xy_coords.shape[:-1])+(winsize, winsize))
    # GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    # GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    
    # GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    GC_scores = sktform.resize(GC_scores,  output_shape=vid.shape[1:]+(winsize,winsize), preserve_range=True, order=1) # this is the problem ? 
    
    return GC_scores


def causal_block_flow_scores_gradient(scores):
    
    """
    scores : M x N x winsize x winsize 
    """
    import numpy as np 
    # collapse the combine all into one into vectors.
    mid = scores.shape[-1]//2
            
    corr_x_direction = -np.apply_over_axes(np.nansum, scores[:,:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, scores[:,:,:,mid+1:], [-1,-2])
    corr_y_direction = -np.apply_over_axes(np.nansum, scores[:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, scores[:,:,mid+1:], [-1,-2])
    intensity = np.nansum(scores.reshape(scores.shape[0], scores.shape[1], -1), axis=-1) 
    
    mean_vector = np.array([np.squeeze(corr_y_direction), 
                            np.squeeze(corr_x_direction)])
    mean_vector = mean_vector * intensity[None,...]
    mean_vector = mean_vector.transpose(1,2,0)
    mean_vector = -mean_vector

    return mean_vector

"""
For use with PCCA!.
"""
def causal_block_flow(vid, cause_fnc, winsize=3, **kwargs):

    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import numpy as np
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    windows_3, xy_coords = _prepare_windows(vid, winsize)

    GC_vectors = []
    
    # for ii in tqdm(np.arange(len(windows_3))):
    for ii in tqdm(np.arange(len(windows_3))):
        
        """
        This should allow usage of all functions that have the same call signature.
        """
        corr_array = cause_fnc(windows_3[ii, :, :], 
                               windows_3[ii, :, :], **kwargs) 
        # corr_array = GC_full_reduced_separate_regress_individual(windows_3[ii], 
        #                                                           windows_3[ii], 
        #                                                               lag=lag, alpha=alpha) #(centre_ref, center)
        corr_array = corr_array.reshape((winsize,winsize))
        mid = corr_array.shape[1]//2
            
        corr_x_direction = -np.nansum(corr_array[:,:mid]) + np.nansum(corr_array[:,mid+1:])
        corr_y_direction = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
        intensity = np.sum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
        
        mean_vector = np.hstack([corr_y_direction, corr_x_direction])
        mean_vector = mean_vector * intensity
        
        GC_vectors.append(mean_vector) # when is this negative? 
            
        
    GC_vectors = np.array(GC_vectors).reshape((xy_coords.shape))
    GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    
    GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    
    
    return GC_vectors


def causal_flow_parallel(vid, cause_fnc, winsize=3, n_jobs=-1, **kwargs):

    from joblib import Parallel, delayed
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import numpy as np
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    windows_3, xy_coords = _prepare_windows(vid, winsize)

    args_list = [(windows_3[ii], winsize, cause_fnc, kwargs) for ii in range(len(windows_3))]
    GC_vectors = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_flow)(a) for a in tqdm(args_list)
    )

    GC_vectors = np.array(GC_vectors).reshape((xy_coords.shape))
    GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])

    return GC_vectors


def causal_flow_scores_parallel(vid, cause_fnc, winsize=3, n_jobs=-1, **kwargs):

    from joblib import Parallel, delayed
    from tqdm import tqdm
    import numpy as np
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    windows_3, xy_coords = _prepare_windows(vid, winsize)

    args_list = [(windows_3[ii], winsize, cause_fnc, kwargs) for ii in range(len(windows_3))]
    GC_scores = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_flow_scores)(a) for a in tqdm(args_list)
    )

    GC_scores = np.array(GC_scores).reshape((xy_coords.shape[:-1])+(winsize, winsize))
    GC_scores = sktform.resize(GC_scores, output_shape=vid.shape[1:]+(winsize, winsize), preserve_range=True, order=1)

    return GC_scores


"""
Parallel version of causal_block_flow. For use with PCCA and other block-based measures.
"""
def causal_block_flow_parallel(vid, cause_fnc, winsize=3, n_jobs=-1, **kwargs):

    from joblib import Parallel, delayed
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import numpy as np
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    windows_3, xy_coords = _prepare_windows(vid, winsize)

    args_list = [(windows_3[ii], winsize, cause_fnc, kwargs) for ii in range(len(windows_3))]
    GC_vectors = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_block_flow)(a) for a in tqdm(args_list)
    )

    GC_vectors = np.array(GC_vectors).reshape((xy_coords.shape))
    GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])

    return GC_vectors


def gaussian_video_pyramid(vid, scales=[1,2,4,8], sigma=1):
    
    import skimage.transform as sktform
    import numpy as np
    import scipy.ndimage as ndimage
    
    # normalise the video if not. 
    vid_ = (vid - vid.min())/(vid.max()-vid.min())
    # vid_ = vid.copy()
    
    if sigma>0:
        vids = [ndimage.gaussian_filter(sktform.resize(vid_, output_shape=(vid_.shape[0], vid_.shape[1]//s, vid_.shape[2]//s), preserve_range=True), sigma=sigma) for s in scales]
    else:
        # no smoothing
        vids = [sktform.resize(vid_, output_shape=(vid_.shape[0], vid_.shape[1]//s, vid_.shape[2]//s), preserve_range=True) for s in scales]
   
    return vids


def laplacian_video_pyramid(vid, scales=[2,4,8], sigma=1):
    
    import skimage.transform as sktform
    import numpy as np
    import ndimage as ndimage
    
    
    # normalise the video if not. 
    # vid_ = (vid - vid.min())/(vid.max()-vid.min())
    
    vids_laplace = []
    vids_blur = [vid_]
        
    for ii in np.arange(len(scales)):   
        s = scales[ii]
        ds_im = sktform.resize(vid_, output_shape=(vid_.shape[0], vid_.shape[1]//s, vid_.shape[2]//s), preserve_range=True)
        ds_im = ndimage.gaussian_filter(ds_im, sigma=1) 
        
        # if ii == len(scales)-1:
        #     vids_laplace.append(ds_im)
        # else:
        diff = vids_blur[-1] - sktform.resize(ds_im, output_shape=vids_blur[-1].shape, preserve_range=True)
        vids_laplace.append(diff)
        
        vids_blur.append(ds_im)
    vids_laplace.append(ds_im)

    return vids_laplace


"""
Confound-aware multi-channel pixel information flow.

For a pair (source, target) -- including source==target (within-channel) --
each sliding window is decomposed into:
  - target_center   : target channel's own center winsize x winsize sub-block (Y to predict)
  - confound_blocks : always-included baseline blocks -- target's own 8 neighbor
                       sub-blocks when source != target (its own spatial dynamics,
                       the "self-channel confound"), plus every channel other than
                       {source, target}'s full 3winsize x 3winsize super-block
  - candidate_positions : the 3x3 grid of blocks being tested for causal contribution --
        source==target: target's own 8 neighbor sub-blocks (center excluded)
        source!=target: source channel's own center + 8 neighbor sub-blocks (all 9)

A confound-aware cause_fnc (e.g. GC_full_reduced_confound, DDC_cause_confound,
Linear_LK_cause_confound, PDC_central_flow_confound, pcca_cause_confound,
nd_xcorr_lag_confound) is called once per candidate position, producing a 3x3
score map, converted to a directional flow vector exactly like causal_block_flow.
"""

_POS_TO_IDX = {
    'tl': (0,0), 'tc': (0,1), 'tr': (0,2),
    'ml': (1,0),              'mr': (1,2),
    'bl': (2,0), 'bc': (2,1), 'br': (2,2),
    'mc': (1,1),
}


def _sub_positions(block, winsize):
    w = winsize
    return {
        'tl': block[0:w,       0:w],       'tc': block[0:w,       w:2*w],     'tr': block[0:w,       2*w:3*w],
        'ml': block[w:2*w,     0:w],       'mc': block[w:2*w,     w:2*w],     'mr': block[w:2*w,     2*w:3*w],
        'bl': block[2*w:3*w,   0:w],       'bc': block[2*w:3*w,   w:2*w],     'br': block[2*w:3*w,   2*w:3*w],
    }


def _confound_and_candidates(windows_dict, ii, winsize, source_idx, target_idx, channel_indices):
    target_positions = _sub_positions(windows_dict[target_idx][ii], winsize)
    target_center = target_positions['mc']

    confound_blocks = []
    if source_idx != target_idx:
        confound_blocks += [v for k, v in target_positions.items() if k != 'mc']
    for ch in channel_indices:
        if ch not in (source_idx, target_idx):
            confound_blocks.append(windows_dict[ch][ii])

    if source_idx == target_idx:
        candidate_positions = {k: v for k, v in target_positions.items() if k != 'mc'}
    else:
        candidate_positions = _sub_positions(windows_dict[source_idx][ii], winsize)

    return target_center, confound_blocks, candidate_positions


def _score_to_vector(corr_array):
    mid = 1  # corr_array is always 3x3
    cx = -np.nansum(corr_array[:, :mid]) + np.nansum(corr_array[:, mid+1:])
    cy = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
    intensity = np.nansum(corr_array)
    return np.hstack([cy, cx]) * intensity


def _worker_causal_flow_multichannel(args):
    target_center, confound_blocks, candidate_positions, cause_fnc_confound, kwargs = args
    corr_array = np.full((3,3), np.nan)
    for pos, block in candidate_positions.items():
        corr_array[_POS_TO_IDX[pos]] = cause_fnc_confound(target_center, confound_blocks, block, **kwargs)
    corr_array = np.nan_to_num(corr_array, nan=0.0)
    return _score_to_vector(corr_array)


def _worker_causal_flow_scores_multichannel(args):
    target_center, confound_blocks, candidate_positions, cause_fnc_confound, kwargs = args
    corr_array = np.full((3,3), np.nan)
    for pos, block in candidate_positions.items():
        corr_array[_POS_TO_IDX[pos]] = cause_fnc_confound(target_center, confound_blocks, block, **kwargs)
    return np.nan_to_num(corr_array, nan=0.0)


def causal_flow_scores_multichannel(vid_channels, cause_fnc_confound, source_idx, target_idx, winsize=3, **kwargs):
    """
    vid_channels : list/tuple of (T,H,W) arrays, one per channel
    Returns raw 3x3 score maps: (H,W,3,3)
    """
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    channel_indices = list(range(len(vid_channels)))

    windows_dict = {}
    xy_coords = None
    for ch in channel_indices:
        w3, xy = _prepare_windows(vid_channels[ch], winsize)
        windows_dict[ch] = w3
        xy_coords = xy

    n_windows = len(windows_dict[target_idx])
    scores = []
    for ii in tqdm(np.arange(n_windows)):
        target_center, confound_blocks, candidate_positions = _confound_and_candidates(
            windows_dict, ii, winsize, source_idx, target_idx, channel_indices)
        corr_array = np.full((3,3), np.nan)
        for pos, block in candidate_positions.items():
            corr_array[_POS_TO_IDX[pos]] = cause_fnc_confound(target_center, confound_blocks, block, **kwargs)
        scores.append(np.nan_to_num(corr_array, nan=0.0))

    vid_shape = vid_channels[target_idx].shape
    scores = np.array(scores).reshape(xy_coords.shape[:-1] + (3,3))
    scores = sktform.resize(scores, output_shape=vid_shape[1:] + (3,3), preserve_range=True, order=1)

    return scores


def causal_flow_multichannel(vid_channels, cause_fnc_confound, source_idx, target_idx, winsize=3, **kwargs):
    """
    vid_channels : list/tuple of (T,H,W) arrays, one per channel
    Returns a directional flow field: (H,W,2)
    """
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    channel_indices = list(range(len(vid_channels)))

    windows_dict = {}
    xy_coords = None
    for ch in channel_indices:
        w3, xy = _prepare_windows(vid_channels[ch], winsize)
        windows_dict[ch] = w3
        xy_coords = xy

    n_windows = len(windows_dict[target_idx])
    vectors = []
    for ii in tqdm(np.arange(n_windows)):
        target_center, confound_blocks, candidate_positions = _confound_and_candidates(
            windows_dict, ii, winsize, source_idx, target_idx, channel_indices)
        corr_array = np.full((3,3), np.nan)
        for pos, block in candidate_positions.items():
            corr_array[_POS_TO_IDX[pos]] = cause_fnc_confound(target_center, confound_blocks, block, **kwargs)
        corr_array = np.nan_to_num(corr_array, nan=0.0)
        vectors.append(_score_to_vector(corr_array))

    vid_shape = vid_channels[target_idx].shape
    vectors = np.array(vectors).reshape(xy_coords.shape)
    vectors[...,0] = ndimage.gaussian_filter(vectors[...,0], sigma=1.)
    vectors[...,1] = ndimage.gaussian_filter(vectors[...,1], sigma=1.)
    vectors = np.dstack([sktform.resize(vectors[...,ch], output_shape=vid_shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])

    return vectors


def causal_flow_scores_multichannel_parallel(vid_channels, cause_fnc_confound, source_idx, target_idx, winsize=3, n_jobs=-1, **kwargs):

    from joblib import Parallel, delayed
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    channel_indices = list(range(len(vid_channels)))

    windows_dict = {}
    xy_coords = None
    for ch in channel_indices:
        w3, xy = _prepare_windows(vid_channels[ch], winsize)
        windows_dict[ch] = w3
        xy_coords = xy

    n_windows = len(windows_dict[target_idx])
    args_list = []
    for ii in range(n_windows):
        target_center, confound_blocks, candidate_positions = _confound_and_candidates(
            windows_dict, ii, winsize, source_idx, target_idx, channel_indices)
        args_list.append((target_center, confound_blocks, candidate_positions, cause_fnc_confound, kwargs))

    scores = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_flow_scores_multichannel)(a) for a in tqdm(args_list)
    )

    vid_shape = vid_channels[target_idx].shape
    scores = np.array(scores).reshape(xy_coords.shape[:-1] + (3,3))
    scores = sktform.resize(scores, output_shape=vid_shape[1:] + (3,3), preserve_range=True, order=1)

    return scores


def causal_flow_multichannel_parallel(vid_channels, cause_fnc_confound, source_idx, target_idx, winsize=3, n_jobs=-1, **kwargs):

    from joblib import Parallel, delayed
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    channel_indices = list(range(len(vid_channels)))

    windows_dict = {}
    xy_coords = None
    for ch in channel_indices:
        w3, xy = _prepare_windows(vid_channels[ch], winsize)
        windows_dict[ch] = w3
        xy_coords = xy

    n_windows = len(windows_dict[target_idx])
    args_list = []
    for ii in range(n_windows):
        target_center, confound_blocks, candidate_positions = _confound_and_candidates(
            windows_dict, ii, winsize, source_idx, target_idx, channel_indices)
        args_list.append((target_center, confound_blocks, candidate_positions, cause_fnc_confound, kwargs))

    vectors = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_worker_causal_flow_multichannel)(a) for a in tqdm(args_list)
    )

    vid_shape = vid_channels[target_idx].shape
    vectors = np.array(vectors).reshape(xy_coords.shape)
    vectors[...,0] = ndimage.gaussian_filter(vectors[...,0], sigma=1.)
    vectors[...,1] = ndimage.gaussian_filter(vectors[...,1], sigma=1.)
    vectors = np.dstack([sktform.resize(vectors[...,ch], output_shape=vid_shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])

    return vectors


def compute_channel_flows(vid_channels, cause_fnc_confound, pairs='all', winsize=3, parallel=False, scores=True, **kwargs):
    """
    Loop over requested (source, target) channel pairs and compute confound-aware
    causal flow for each.

    vid_channels : list/tuple of (T,H,W) arrays, one per channel
    pairs : 'all' (every source/target combination, C^2 fields),
            'within' (source==target only), 'cross' (source!=target only),
            or an explicit list of (source_idx, target_idx) tuples
    scores : if True, use the *_scores_* drivers (returns (H,W,3,3) score maps);
             if False, use the vector drivers (returns (H,W,2) flow fields)

    Returns dict {(source_idx, target_idx): flow_array}
    """
    n = len(vid_channels)
    if pairs == 'all':
        pair_list = [(s,t) for s in range(n) for t in range(n)]
    elif pairs == 'within':
        pair_list = [(c,c) for c in range(n)]
    elif pairs == 'cross':
        pair_list = [(s,t) for s in range(n) for t in range(n) if s != t]
    else:
        pair_list = list(pairs)

    if scores:
        driver_fn = causal_flow_scores_multichannel_parallel if parallel else causal_flow_scores_multichannel
    else:
        driver_fn = causal_flow_multichannel_parallel if parallel else causal_flow_multichannel

    results = {}
    for (s, t) in pair_list:
        results[(s, t)] = driver_fn(vid_channels, cause_fnc_confound, s, t, winsize=winsize, **kwargs)

    return results


"""
Separate, simpler joint-regression multi-channel drivers for DDC and LK,
kept independent from the confound-aware family above for A/B testing.

DDC_cause/Linear_LK_cause get their directional signal from a specific
mechanism: they stack the *same* block twice ("original" and "copy",
identical data) and read the resulting cross-block's diagonal -- the
self-block diagonal alone (each pixel's own autoregression coefficient)
carries no directional information. So instead of decomposing into
neighbor positions, these treat the C channels themselves as the "neighbors":
flatten every channel's center winsize x winsize block once as an "original"
and once again as a "copy", stack all 2*C blocks into one design matrix, and
fit ONE differential_covariance/Linear_LK regression per window. Every
ordered channel pair's directional coefficients are then read directly out
of that single fit's coefficient matrix (target channel's "original" rows
vs. source channel's "copy" columns) -- a joint fit across every channel
already conditions each pairwise coefficient on all the others, so no
separate confound bookkeeping is needed.

With C=1 the two-copy stack is exactly [ch0, ch0_copy] -- the same stack
causal_flow(ch0, DDC_cause/Linear_LK_cause, ...) already builds -- so the
(0,0) result reduces identically to today's single-channel output.
"""


def _joint_ddc_corr_array(originals_flat, copies_flat, winsize, source_idx, target_idx, eps, alpha):
    from InfoFlow.DDC_flow import differential_covariance

    ws2 = winsize * winsize
    C = len(originals_flat)
    stacked = np.vstack(originals_flat + copies_flat)

    W_ = differential_covariance(stacked, eps=eps, alpha=alpha)

    target_rows = slice(target_idx * ws2, (target_idx + 1) * ws2)
    source_cols = slice(C * ws2 + source_idx * ws2, C * ws2 + (source_idx + 1) * ws2)
    W_sub = W_[target_rows, source_cols]

    return np.diag(W_sub).reshape(winsize, winsize)


def _joint_lk_corr_array(originals_flat, copies_flat, winsize, source_idx, target_idx, eps):
    from InfoFlow.LK_flow import Linear_LK

    ws2 = winsize * winsize
    C = len(originals_flat)
    stacked = np.vstack(originals_flat + copies_flat)

    # Linear_LK_cause does not forward alpha to Linear_LK either -- matched here for exact equivalence at C=1.
    W_, C_ = Linear_LK(stacked, eps=eps)

    target_rows = slice(target_idx * ws2, (target_idx + 1) * ws2)
    source_cols = slice(C * ws2 + source_idx * ws2, C * ws2 + (source_idx + 1) * ws2)
    W_sub = W_[target_rows, source_cols]
    C_sub = C_[target_rows, source_cols]
    C_diag_target = np.diag(C_)[target_rows]

    score = np.diag(W_sub) * np.sqrt(np.abs(np.diag(C_sub)) / (C_diag_target + eps) + eps)

    return score.reshape(winsize, winsize)


def _joint_center_blocks(windows_dict, ii, winsize, n_channels):
    ws2 = winsize * winsize
    originals = [windows_dict[c][ii, winsize:2*winsize, winsize:2*winsize].reshape(ws2, -1) for c in range(n_channels)]
    copies = [o.copy() for o in originals]
    return originals, copies


def causal_flow_scores_multichannel_joint_ddc(vid_channels, winsize=3, eps=1e-12, alpha=1e-2):
    """
    Returns dict {(source_idx, target_idx): (H,W,winsize,winsize)} for all C^2 pairs,
    computed from one differential_covariance fit per window.
    """
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    scores = {pair: [] for pair in pair_list}

    for ii in tqdm(np.arange(n_windows)):
        originals, copies = _joint_center_blocks(windows_dict, ii, winsize, n_channels)
        for (s, t) in pair_list:
            scores[(s, t)].append(_joint_ddc_corr_array(originals, copies, winsize, s, t, eps, alpha))

    results = {}
    for pair in pair_list:
        arr = np.array(scores[pair]).reshape(xy_coords.shape[:-1] + (winsize, winsize))
        vid_shape = vid_channels[pair[1]].shape
        results[pair] = sktform.resize(arr, output_shape=vid_shape[1:] + (winsize, winsize), preserve_range=True, order=1)

    return results


def causal_flow_multichannel_joint_ddc(vid_channels, winsize=3, eps=1e-12, alpha=1e-2):
    """
    Returns dict {(source_idx, target_idx): (H,W,2)} for all C^2 pairs,
    computed from one differential_covariance fit per window.
    """
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    vectors = {pair: [] for pair in pair_list}
    mid = winsize // 2

    for ii in tqdm(np.arange(n_windows)):
        originals, copies = _joint_center_blocks(windows_dict, ii, winsize, n_channels)
        for (s, t) in pair_list:
            corr_array = _joint_ddc_corr_array(originals, copies, winsize, s, t, eps, alpha)
            cx = -np.nansum(corr_array[:, :mid]) + np.nansum(corr_array[:, mid+1:])
            cy = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
            intensity = np.nansum(corr_array)
            mean_vector = np.hstack([cy, cx]) * intensity
            vectors[(s, t)].append(-mean_vector)

    results = {}
    for pair in pair_list:
        vid_shape = vid_channels[pair[1]].shape
        vec = np.array(vectors[pair]).reshape(xy_coords.shape)
        vec[..., 0] = ndimage.gaussian_filter(vec[..., 0], sigma=1.)
        vec[..., 1] = ndimage.gaussian_filter(vec[..., 1], sigma=1.)
        vec = np.dstack([sktform.resize(vec[..., ch], output_shape=vid_shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
        results[pair] = vec

    return results


def causal_flow_scores_multichannel_joint_lk(vid_channels, winsize=3, eps=1e-12):
    """
    Returns dict {(source_idx, target_idx): (H,W,winsize,winsize)} for all C^2 pairs,
    computed from one Linear_LK fit per window.
    """
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    scores = {pair: [] for pair in pair_list}

    for ii in tqdm(np.arange(n_windows)):
        originals, copies = _joint_center_blocks(windows_dict, ii, winsize, n_channels)
        for (s, t) in pair_list:
            scores[(s, t)].append(_joint_lk_corr_array(originals, copies, winsize, s, t, eps))

    results = {}
    for pair in pair_list:
        arr = np.array(scores[pair]).reshape(xy_coords.shape[:-1] + (winsize, winsize))
        vid_shape = vid_channels[pair[1]].shape
        results[pair] = sktform.resize(arr, output_shape=vid_shape[1:] + (winsize, winsize), preserve_range=True, order=1)

    return results


def causal_flow_multichannel_joint_lk(vid_channels, winsize=3, eps=1e-12):
    """
    Returns dict {(source_idx, target_idx): (H,W,2)} for all C^2 pairs,
    computed from one Linear_LK fit per window.
    """
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    vectors = {pair: [] for pair in pair_list}
    mid = winsize // 2

    for ii in tqdm(np.arange(n_windows)):
        originals, copies = _joint_center_blocks(windows_dict, ii, winsize, n_channels)
        for (s, t) in pair_list:
            corr_array = _joint_lk_corr_array(originals, copies, winsize, s, t, eps)
            cx = -np.nansum(corr_array[:, :mid]) + np.nansum(corr_array[:, mid+1:])
            cy = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
            intensity = np.nansum(corr_array)
            mean_vector = np.hstack([cy, cx]) * intensity
            vectors[(s, t)].append(-mean_vector)

    results = {}
    for pair in pair_list:
        vid_shape = vid_channels[pair[1]].shape
        vec = np.array(vectors[pair]).reshape(xy_coords.shape)
        vec[..., 0] = ndimage.gaussian_filter(vec[..., 0], sigma=1.)
        vec[..., 1] = ndimage.gaussian_filter(vec[..., 1], sigma=1.)
        vec = np.dstack([sktform.resize(vec[..., ch], output_shape=vid_shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
        results[pair] = vec

    return results


"""
Separate, simpler joint-regression multi-channel driver for correlation
(nd_xcorr_lag), kept independent from the confound-aware family for A/B
testing -- same overall spirit as the DDC/LK joint drivers above, but
nd_xcorr_lag is structurally different: it already produces a genuine
winsize x winsize spatial correlation map via scipy.signal.correlate on two
blocks directly, with no flatten/diagonal-trick needed.

Stacking all C channels' blocks and correlating the stack against ITSELF
doesn't work here: scipy.signal.correlate's channel-axis output is indexed
by offset (target_channel - source_channel), not by the actual pair, so
every within-channel comparison collapses onto the same offset=0 slot
regardless of which channel, and cross-channel offsets are shared by more
than one pair once C>2.

Instead, for a given target channel, its own block is used as a size-1
"kernel" along the channel axis against a "volume" stacking all C channels.
A size-1 axis can't slide, so correlate degenerates on that axis to an
independent per-channel-slot correlation -- one call yields the (y,x,time)
correlation between the target and every actual source channel, indexed
by real channel index, with no conflation at any C.
"""


def _joint_corr_score(windows_dict, ii, winsize, n_channels, target_idx, lag, mode, demean):
    import numpy as np
    from scipy.signal import correlate

    full_stack = np.stack(
        [windows_dict[c][ii, winsize:2*winsize, winsize:2*winsize] for c in range(n_channels)],
        axis=2,
    )  # (winsize, winsize, C, T)

    volume = full_stack[..., :full_stack.shape[-1] - lag].copy()          # all channels' past
    template = full_stack[:, :, target_idx:target_idx+1, lag:].copy()     # target channel's future

    if demean:
        volume = volume - volume.mean(axis=-1, keepdims=True)
        template = template - template.mean(axis=-1, keepdims=True)

    corr = correlate(volume, template, mode=mode)
    corr = np.nanmax(corr, axis=-1)  # reduce over time, exactly like nd_xcorr_lag

    return corr  # (winsize, winsize, C) -- corr[:, :, s] is the (source=s, target=target_idx) map


def causal_flow_scores_multichannel_joint_corr(vid_channels, winsize=3, lag=1, mode='same', demean=False):
    """
    Returns dict {(source_idx, target_idx): (H,W,winsize,winsize)} for all C^2 pairs,
    computed from C correlate() calls per window (one per target channel).
    """
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    scores = {pair: [] for pair in pair_list}

    for ii in tqdm(np.arange(n_windows)):
        for t in range(n_channels):
            corr = _joint_corr_score(windows_dict, ii, winsize, n_channels, t, lag, mode, demean)
            for s in range(n_channels):
                scores[(s, t)].append(corr[:, :, s])

    results = {}
    for pair in pair_list:
        arr = np.array(scores[pair]).reshape(xy_coords.shape[:-1] + (winsize, winsize))
        vid_shape = vid_channels[pair[1]].shape
        results[pair] = sktform.resize(arr, output_shape=vid_shape[1:] + (winsize, winsize), preserve_range=True, order=1)

    return results


def causal_flow_multichannel_joint_corr(vid_channels, winsize=3, lag=1, mode='same', demean=False):
    """
    Returns dict {(source_idx, target_idx): (H,W,2)} for all C^2 pairs,
    computed from C correlate() calls per window (one per target channel).
    """
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    vectors = {pair: [] for pair in pair_list}
    mid = winsize // 2

    for ii in tqdm(np.arange(n_windows)):
        for t in range(n_channels):
            corr = _joint_corr_score(windows_dict, ii, winsize, n_channels, t, lag, mode, demean)
            for s in range(n_channels):
                corr_array = corr[:, :, s]
                cx = -np.nansum(corr_array[:, :mid]) + np.nansum(corr_array[:, mid+1:])
                cy = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
                intensity = np.nansum(corr_array)
                mean_vector = np.hstack([cy, cx]) * intensity
                vectors[(s, t)].append(-mean_vector)

    results = {}
    for pair in pair_list:
        vid_shape = vid_channels[pair[1]].shape
        vec = np.array(vectors[pair]).reshape(xy_coords.shape)
        vec[..., 0] = ndimage.gaussian_filter(vec[..., 0], sigma=1.)
        vec[..., 1] = ndimage.gaussian_filter(vec[..., 1], sigma=1.)
        vec = np.dstack([sktform.resize(vec[..., ch], output_shape=vid_shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
        results[pair] = vec

    return results


"""
Separate, simpler joint-regression multi-channel driver for GC, kept
independent from the confound-aware family for A/B testing.

GC_full_reduced_separate_regress_individual has a real bug (see
InfoFlow/gc_flow.py's commented-out lines 39-41): its full model uses only
img2's own history+contemporaneous value, never including img1's (target's)
own lag history, so the reduced model's regressors aren't a subset of the
full model's -- logF-logL isn't a valid nested conditional-causality delta.

This driver fixes that AND generalizes to C channels at once: build one
combined block spanning every channel's own winsize x winsize center
sub-block (no duplicate "copy" trick needed, unlike DDC/LK -- GC's
reduced-vs-full comparison already produces a meaningful delta from a single
combined block, same as today's single-channel usage with img1 is img2).

Per window: one shared REDUCED fit (Y = every channel's own future pixels;
X_reduced = the combined block's own lagged history, i.e. baseline already
includes every channel's own history) plus one FULL fit per candidate source
channel s (X_full_s = X_reduced -- restoring the nesting fix -- plus only
channel s's own contemporaneous value). Since Y spans every channel,
delta_s reshaped to (C,winsize,winsize) gives every (source=s,target=t)
pair's map from just C+1 Ridge fits per window, not C^2.
"""


def causal_flow_scores_multichannel_joint_gc(vid_channels, winsize=3, lag=1, alpha=.1):
    """
    Returns dict {(source_idx, target_idx): (H,W,winsize,winsize)} for all C^2 pairs,
    computed from one shared reduced fit + C full fits (one per candidate source) per window.
    """
    from sklearn.linear_model import Ridge
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)
    ws2 = winsize * winsize

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    scores = {pair: [] for pair in pair_list}

    for ii in tqdm(np.arange(n_windows)):
        blocks = [windows_dict[c][ii, winsize:2*winsize, winsize:2*winsize].reshape(ws2, -1) for c in range(n_channels)]
        combined_t = np.vstack(blocks).T  # (T, C*ws2)

        Y = combined_t[lag:]
        X_reduced = np.hstack([combined_t[lag-ll:-ll] for ll in range(1, lag+1)])

        clf = Ridge(alpha=alpha)
        clf.fit(X_reduced, Y)
        logL = np.log(np.var(Y - clf.predict(X_reduced), axis=0))

        for s in range(n_channels):
            candidate_now = combined_t[lag:, s*ws2:(s+1)*ws2]
            X_full = np.hstack([X_reduced, candidate_now])

            clf_full = Ridge(alpha=alpha)
            clf_full.fit(X_full, Y)
            logF = np.log(np.var(Y - clf_full.predict(X_full), axis=0))

            delta = (logF - logL).reshape(n_channels, winsize, winsize)
            for t in range(n_channels):
                scores[(s, t)].append(delta[t])

    results = {}
    for pair in pair_list:
        arr = np.array(scores[pair]).reshape(xy_coords.shape[:-1] + (winsize, winsize))
        vid_shape = vid_channels[pair[1]].shape
        results[pair] = sktform.resize(arr, output_shape=vid_shape[1:] + (winsize, winsize), preserve_range=True, order=1)

    return results


def causal_flow_multichannel_joint_gc(vid_channels, winsize=3, lag=1, alpha=.1):
    """
    Returns dict {(source_idx, target_idx): (H,W,2)} for all C^2 pairs,
    computed from one shared reduced fit + C full fits (one per candidate source) per window.
    """
    from sklearn.linear_model import Ridge
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    import skimage.transform as sktform

    winsize = _snap_odd_winsize(winsize)
    n_channels = len(vid_channels)
    ws2 = winsize * winsize

    windows_dict = {}
    xy_coords = None
    for c, vid in enumerate(vid_channels):
        w3, xy = _prepare_windows(vid, winsize)
        windows_dict[c] = w3
        xy_coords = xy

    n_windows = len(windows_dict[0])
    pair_list = [(s, t) for s in range(n_channels) for t in range(n_channels)]
    vectors = {pair: [] for pair in pair_list}
    mid = winsize // 2

    for ii in tqdm(np.arange(n_windows)):
        blocks = [windows_dict[c][ii, winsize:2*winsize, winsize:2*winsize].reshape(ws2, -1) for c in range(n_channels)]
        combined_t = np.vstack(blocks).T  # (T, C*ws2)

        Y = combined_t[lag:]
        X_reduced = np.hstack([combined_t[lag-ll:-ll] for ll in range(1, lag+1)])

        clf = Ridge(alpha=alpha)
        clf.fit(X_reduced, Y)
        logL = np.log(np.var(Y - clf.predict(X_reduced), axis=0))

        for s in range(n_channels):
            candidate_now = combined_t[lag:, s*ws2:(s+1)*ws2]
            X_full = np.hstack([X_reduced, candidate_now])

            clf_full = Ridge(alpha=alpha)
            clf_full.fit(X_full, Y)
            logF = np.log(np.var(Y - clf_full.predict(X_full), axis=0))

            delta = (logF - logL).reshape(n_channels, winsize, winsize)
            for t in range(n_channels):
                corr_array = delta[t]
                cx = -np.nansum(corr_array[:, :mid]) + np.nansum(corr_array[:, mid+1:])
                cy = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
                intensity = np.nansum(corr_array)
                mean_vector = np.hstack([cy, cx]) * intensity
                vectors[(s, t)].append(-mean_vector)

    results = {}
    for pair in pair_list:
        vid_shape = vid_channels[pair[1]].shape
        vec = np.array(vectors[pair]).reshape(xy_coords.shape)
        vec[..., 0] = ndimage.gaussian_filter(vec[..., 0], sigma=1.)
        vec[..., 1] = ndimage.gaussian_filter(vec[..., 1], sigma=1.)
        vec = np.dstack([sktform.resize(vec[..., ch], output_shape=vid_shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
        results[pair] = vec

    return results


if __name__=="__main__":

    # import pyinform
    import numpy as np
    import scipy.io
    #import random
    # from pyinform import transfer_entropy 
    from scipy.ndimage import gaussian_filter
    from matplotlib import pyplot as plt
    # from PyIF import te_compute as te
    # import te_compute as te
    ## import matCellRatioT
    from skimage.transform import rescale, resize, downscale_local_mean
    from skimage import color
    from tqdm import tqdm 
    import skimage.transform as sktform
    import skimage.util as skutil 
    import scipy.ndimage as ndimage 

    """
    Imports of various flow functions. 
    """
    # from InfoFlow.gc_flow import GC_full_reduced_separate_regress_individual
    # from InfoFlow.DDC_flow import DDC_cause
    # from InfoFlow.pdc_dtf_flow import PDC_central_flow
    # from InfoFlow.pcca_flow import pcca_cause_block
    # from InfoFlow.correlation_flow import nd_xcorr_lag
    # from InfoFlow.LK_flow import Linear_LK_cause
    from gc_flow import GC_full_reduced_separate_regress_individual
    from DDC_flow import DDC_cause
    from pdc_dtf_flow import PDC_central_flow
    from pcca_flow import pcca_cause_block
    from correlation_flow import nd_xcorr_lag
    from optical_flow import extract_optflow
    from LK_flow import Linear_LK_cause
    from flow_vis import flow_to_color
    
    from dynamic_image import _compute_dynamic_image
    
    
    """
    Write a script to get the dynamic image. 
    """
    

    #def rgb2gray(rgb):
    
    #   r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    #    return gray
    # myVid = read_video_cv2(r'../821-10_l.mov') #works ! # weird ? 
    # myVid = read_video_cv2(r'../9-19_l.mov') #works !
    # myVid =  read_video_cv2(r'../001-0436.avi') # works!
    myVid = read_video_cv2(r'../3687-18_70.mov') #### spurious arrows present for the LK flow. --- is this due to the implementation of the estimation? # can we instead use properly the maximum likelihood estimate? 
    # myVid = read_video_cv2(r'../620-72_l.mov')
    # myVid = read_video_cv2(r'../637-147_l.mov')
    # myVid = read_video_cv2(r'341-46_l.mov')
    # myVid = read_video_cv2(r'7399-1_70.mov')
    # myVid = read_video_cv2(r'2082-3_70.mov')
    # myVid = read_video_cv2(r'1174-6_70.mov')
    # myVid = read_video_cv2(r'879-38_l.mov')
    
    # myVid = read_video_cv2(r'965-126_l.mov')
    # myVid = read_video_cv2(r'621-3_l.mov')
    
    # import imageio
    # # import numpy as np
    # vid = imageio.get_reader(r'7399-1_70.mov',  'ffmpeg')
    # myVid = np.array([im for im in vid.iter_data()], dtype=np.uint8)
    myVid = color.rgb2gray(myVid)
    
    dyn_image_Vid = _compute_dynamic_image(myVid[...,None])
    
    plt.figure(figsize=(5,5))
    plt.title('Dynamic Image')
    plt.imshow(dyn_image_Vid, cmap='gray')
    plt.show()

    plt.figure(figsize=(5,5))
    plt.title('Mean Image')
    plt.imshow(np.nanmean(myVid, axis=0), cmap='gray')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.title('Max Image')
    plt.imshow(np.nanmax(myVid, axis=0), cmap='gray')
    plt.show()
    
    
    """
    Are there other ways to downsample? - do we need to add smoothing after each bilinear interpolation downsample? 
    """
    # myVid = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1], myVid.shape[2]), preserve_range=True)
    # myVid2 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//2, myVid.shape[2]//2), preserve_range=True)
    # myVid4 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//4, myVid.shape[2]//4), preserve_range=True)
    # myVid8 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//8, myVid.shape[2]//8), preserve_range=True)
    
    myVid, myVid2, myVid4, myVid8 = gaussian_video_pyramid(myVid, scales=[1,2,4,8], sigma=1)
    # myVid, myVid2, myVid4, myVid8 = laplacian_video_pyramid(myVid, scales=[2,4,8], sigma=1)
    
    
# =============================================================================
#     1. extract all windowed - flat
# =============================================================================
    
    # # this is normal GC. 
    # GC_vectors = causal_flow(myVid, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    # GC_vectors2 = causal_flow(myVid2, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    # GC_vectors4 = causal_flow(myVid4, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    # GC_vectors8 = causal_flow(myVid8, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    
    GC_vectors = causal_flow_scores(myVid, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    GC_vectors2 = causal_flow_scores(myVid2, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    GC_vectors4 = causal_flow_scores(myVid4, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    GC_vectors8 = causal_flow_scores(myVid8, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    
    # # this is DDC
    # GC_vectors = causal_flow(myVid, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors2 = causal_flow(myVid2, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors4 = causal_flow(myVid4, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors8 = causal_flow(myVid8, DDC_cause, eps=1e-12, alpha=1e-2)
    
    # GC_vectors = causal_flow_scores(myVid, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors2 = causal_flow_scores(myVid2, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors4 = causal_flow_scores(myVid4, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors8 = causal_flow_scores(myVid8, DDC_cause, eps=1e-12, alpha=1e-2)
    
    # # # linear LK 
    # # # # GC = Linear_LK_cause
    # GC_vectors = causal_flow(myVid, Linear_LK_cause, eps=1e-12)
    # GC_vectors2 = causal_flow(myVid2, Linear_LK_cause, eps=1e-12)
    # GC_vectors4 = causal_flow(myVid4, Linear_LK_cause, eps=1e-12)
    # GC_vectors8 = causal_flow(myVid8, Linear_LK_cause, eps=1e-12)
    
    # GC_vectors = causal_flow_scores(myVid, Linear_LK_cause, eps=1e-12)
    # GC_vectors2 = causal_flow_scores(myVid2, Linear_LK_cause, eps=1e-12)
    # GC_vectors4 = causal_flow_scores(myVid4, Linear_LK_cause, eps=1e-12)
    # GC_vectors8 = causal_flow_scores(myVid8, Linear_LK_cause, eps=1e-12)
    
#     # this is the PDC --- very very slow!. 
#     # print('computing original resolution ...')
#     # GC_vectors = causal_flow(myVid, PDC_central_flow, lag=5, alpha=1e-2) # this seems very slow? 2hr!!!! 
#     # print('computing 2x downsample resolution ...')
#     # GC_vectors2 = causal_flow(myVid2, PDC_central_flow, lag=5, alpha=1e-2)
#     # print('computing 4x downsample resolution ...')
#     # GC_vectors4 = causal_flow(myVid4, PDC_central_flow, lag=5, alpha=1e-2)
#     # print('computing 8x downsample resolution ...')
#     # GC_vectors8 = causal_flow(myVid8, PDC_central_flow, lag=5, alpha=1e-2)
    
    
#     # # # this is the correlation flow
#     # GC_vectors = causal_flow(myVid, nd_xcorr_lag, lag=1, mode='same')
#     # GC_vectors2 = causal_flow(myVid2, nd_xcorr_lag, lag=1, mode='same')
#     # GC_vectors4 = causal_flow(myVid4, nd_xcorr_lag, lag=1, mode='same')
#     # GC_vectors8 = causal_flow(myVid8, nd_xcorr_lag, lag=1, mode='same')
    
    
#     # now we need to do e.g. 
    
    
#     # # # PCCA flow. 
#     # # GC_vectors = causal_block_flow(myVid, pcca_cause_block, 
#     # #                                block_size=3,
#     # #                                k=1, 
#     # #                                m=3, 
#     # #                                eta_xt=5e-4, 
#     # #                                eta_yt=5e-4,
#     # #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
    
#     # GC_vectors2 = causal_block_flow(myVid2, pcca_cause_block, 
#     #                                block_size=3,
#     #                                k=1, 
#     #                                m=3, 
#     #                                eta_xt=5e-4, 
#     #                                eta_yt=5e-4,
#     #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
    
#     # GC_vectors4 = causal_block_flow(myVid4, pcca_cause_block, 
#     #                                block_size=3,
#     #                                k=1, 
#     #                                m=3, 
#     #                                eta_xt=5e-4, 
#     #                                eta_yt=5e-4,
#     #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
    
#     # GC_vectors8 = causal_block_flow(myVid8, pcca_cause_block, 
#     #                                block_size=3,
#     #                                k=1, 
#     #                                m=3, 
#     #                                eta_xt=5e-4, 
#     #                                eta_yt=5e-4,
#     #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
#     GC_vectors = GC_vectors 
#     GC_vectors2_resize = np.dstack([sktform.resize(GC_vectors2[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
#     GC_vectors4_resize = np.dstack([sktform.resize(GC_vectors4[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
#     GC_vectors8_resize = np.dstack([sktform.resize(GC_vectors8[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    GC_vectors = GC_vectors 
    GC_vectors2_resize = sktform.resize(GC_vectors2, output_shape=myVid.shape[1:]+GC_vectors2.shape[-2:], preserve_range=True, order=1)
    GC_vectors4_resize = sktform.resize(GC_vectors4, output_shape=myVid.shape[1:]+GC_vectors4.shape[-2:], preserve_range=True, order=1)
    GC_vectors8_resize = sktform.resize(GC_vectors8, output_shape=myVid.shape[1:]+GC_vectors8.shape[-2:], preserve_range=True, order=1)
#     # what is the best way to combine? 
#     # GC_vectors_combine = 1./4*(GC_vectors + GC_vectors2_resize + GC_vectors4_resize + GC_vectors8_resize)
#     # GC_vectors_combine = 1*GC_vectors + 2*GC_vectors2_resize +4* GC_vectors4_resize + 8*GC_vectors8_resize
#     # GC_vectors_combine = GC_vectors_combine / ( 1 + 2 + 4 + 8)
    
# # =============================================================================
# #     2. is this really the best way to combine ? 
# # =============================================================================
    GC_vectors_combine = np.nanmean( np.array([GC_vectors, GC_vectors2_resize, GC_vectors4_resize, GC_vectors8_resize]), axis=0) # combine 
#     # GC_vectors_combine = np.nanmean( np.array([GC_vectors2_resize, GC_vectors4_resize, GC_vectors8_resize]), axis=0)
    
    # # collapse the combine all into one into vectors.
    # mid = GC_vectors_combine.shape[-1]//2
            
    # corr_x_direction = -np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,:,mid+1:], [-1,-2])
    # corr_y_direction = -np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,mid+1:], [-1,-2])
    # intensity = np.nansum(GC_vectors_combine.reshape(GC_vectors_combine.shape[0], GC_vectors_combine.shape[1], -1), axis=-1) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
    
    # mean_vector = np.array([np.squeeze(corr_y_direction), 
    #                          np.squeeze(corr_x_direction)])
    # mean_vector = mean_vector * intensity[None,...]
    # mean_vector = mean_vector.transpose(1,2,0)
    # mean_vector = -mean_vector
    
    mean_vector = causal_block_flow_scores_gradient(GC_vectors_combine)


    xy_coords = np.indices(myVid.shape[1:]); xy_coords=xy_coords.transpose(1,2,0)
    xy_coords = xy_coords[...,::-1]
    
    sampling = 8
#     plt.figure(figsize=(15,15))
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors[::sampling,::sampling,1],  # x 
#                 -GC_vectors[::sampling,::sampling,0]) # y 
#     plt.show()
    
    
#     plt.figure(figsize=(15,15))
#     plt.title('2x')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors2_resize[::sampling,::sampling,1],  # x 
#                 -GC_vectors2_resize[::sampling,::sampling,0]) # y 
#     plt.show()
    
    
#     plt.figure(figsize=(15,15))
#     plt.title('4x')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors4_resize[::sampling,::sampling,1],  # x 
#                 -GC_vectors4_resize[::sampling,::sampling,0]) # y 
#     plt.show()
    
#     plt.figure(figsize=(15,15))
#     plt.title('8x')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors8_resize[::sampling,::sampling,1],  # x 
#                 -GC_vectors8_resize[::sampling,::sampling,0]) # y 
#     plt.show()
    
#     plt.figure(figsize=(15,15))
#     plt.title('Combine')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors_combine[::sampling,::sampling,1],  # x 
#                 -GC_vectors_combine[::sampling,::sampling,0]) # y 
#     plt.show()


    plt.figure(figsize=(15,15))
    plt.title('Combine')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                mean_vector[::sampling,::sampling,1],  # x 
                -mean_vector[::sampling,::sampling,0]) # y 
    plt.show()
    
    
#     # plt.figure(figsize=(15,15))
#     # # plt.imshow(myVid[1])
#     # plt.imshow(frame_a_[...,0])
#     # plt.plot(x.ravel(), 
#     #          y.ravel(), 'k.')
#     # plt.show()
    
    
    
    mean_flow_color = flow_to_color(mean_vector[...,::-1])
    
    plt.figure(figsize=(5,5))
    plt.imshow(mean_flow_color)
    plt.show()
    
    
    # """
    # This is correct!..... 
    # """
    # optical_flow_params = dict(pyr_scale=0.5, levels=1, winsize=3, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
     
    # vid_flow = extract_optflow(255*myVid[:], 
    #                             optical_flow_params, 
    #                             rescale_intensity=False, 
    #                             intensity_range=[2,98])
    
    # mean_opt_flow_color = flow_to_color(vid_flow.mean(axis=0))
    
    # plt.figure(figsize=(5,5))
    # plt.imshow(mean_opt_flow_color)
    # plt.show()
    
    
    optical_flow_params = dict(pyr_scale=0.5, levels=4, winsize=5, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
     
    vid_flow = extract_optflow(255*myVid[:], 
                                optical_flow_params, 
                                rescale_intensity=False, 
                                intensity_range=[2,98])
    
    mean_opt_flow_color = flow_to_color(vid_flow.mean(axis=0))
    
    plt.figure(figsize=(5,5))
    plt.imshow(mean_opt_flow_color)
    plt.show()
    
    
    plt.figure(figsize=(15,15))
    plt.title('mean optical flow')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
               xy_coords[::sampling,::sampling,1], 
                np.nanmean(vid_flow, axis=0)[::sampling,::sampling,0],  # x 
                -np.nanmean(vid_flow, axis=0)[::sampling,::sampling,1]) # y 
    plt.show()
    
    
    plt.figure(figsize=(5,5))
    plt.subplot(121)
    plt.title('Optical flow')
    plt.imshow(mean_opt_flow_color)
    plt.subplot(122)
    plt.title('linear GC')
    plt.imshow(mean_flow_color)
    plt.show()
    
    
    
#     # # myVid = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1], myVid.shape[2]), pre serve_range=True)

#     # myVid = (myVid - np.mean(myVid)) / np.std(myVid)
#     # test = sliding_window_array(myVid[0], window_size = 64, overlap = 32)
#     # test_2 = moving_window_array(myVid[0], window_size = 64, overlap = 32)
#     """
#     Determine the padding based on winsize + striding. # winsize should determine this.
#     """
#     # winsize = 5 # take only normal sizes # block may not be good for this ... 
#     winsize = 3
#     stride = winsize # this is fine. 
    
#     # added settings to allow assessment of length of video. ---> this works!. wow.! 
#     start = 0
#     end = start + len(myVid)
    
#     frame_a = myVid[start:end].transpose(1,2,0)
#     frame_b = myVid[start:end].transpose(1,2,0)
    
#     # frame_a = frame_a.transpose(1,2,0)
#     # frame_b = frame_b.transpose(1,2,0)
#     frame_a_ = np.pad(frame_a, [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
#     frame_b_ = np.pad(frame_b, [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
#     # frame_a_ = np.pad(frame_a, [[winsize,winsize], [winsize,winsize], [0,0]], mode='edge')
#     # frame_b_ = np.pad(frame_b, [[winsize,winsize], [winsize,winsize], [0,0]], mode='edge')
    
#     # frame_a_ = np.pad(frame_a, [[winsize,winsize], [winsize,winsize]], mode='reflect')
#     # frame_b_ = np.pad(frame_b, [[winsize,winsize], [winsize,winsize]], mode='edge')
    
#     M,N = frame_a.shape[:2]
#     # figure out the number of window iterations to take. 
#     row_indices = np.arange(0, M-winsize, stride) + winsize # these are the central coordinates. 
#     col_indices = np.arange(0, N-winsize, stride) + winsize 
    
#     # now we can iterate and crop out. 
#     out_vect = np.zeros((len(row_indices), len(col_indices), 2))
#     xy_coords = np.zeros((len(row_indices), len(col_indices), 2))
    
#     def xcorr2(A,B, norm=True):
#         from scipy.signal import correlate2d
#         if norm:
#             # A_ = (A-np.nanmean(A, axis=1)[None,:] / (np.nanstd(A, axis=1))[None,:])
#             # B_ = (B-np.nanmean(B, axis=1)[None,:] / (np.nanstd(B, axis=1))[None,:])
#             A_ = (A-np.nanmean(A)) / (np.nanstd(A)*np.prod(A.shape[:2]))
#             B_ = (B-np.nanmean(B)) / (np.nanstd(B))
#         else:
#             A_ = A.copy()
#             B_ = B.copy()
            
#         # compute the dot product. 
#         return correlate2d(A_, B_, mode='same', boundary='fill')
    
    
#     def normxcorr2(A,B, norm=True):
#         from scipy.signal import correlate2d
#         from skimage.feature import match_template
        
#         # A_ = A-np.nanmean(A)
#         # B_ = B-np.nanmean(B)
        
#         # x = match_template(A,B, pad_input=True)
#         # return x 
#         # # return np.max(x)
#         # # return x[A.shape[0]//2, A.shape[1]//2]
#         # # return float(correlate2d(A-A.mean(), B-B.mean(), mode='valid'))
#         # # A_ = (A-np.nanmean(A)) / (np.nanstd(A)*np.prod(A.shape[:2]))
#         # # # B_ = (B-np.nanmean(B)) / (np.nanstd(B))
#         A_ = A-np.nanmean(A)
#         B_ = B-np.nanmean(B)
        
#         # # return float(np.max(correlate2d(A_,B_, mode='same')))
#         return correlate2d(A_,B_, mode='same')
#         # # return A_.ravel().dot(B_.ravel())
    
#     """
#     Directionality is weird? 
#     """
#     # is this sliding correct? 
#     # def granger_naive(img1, img2):
#     #     from CausalCalculator import CausalCalculator
#     #     ## zero_pad # we scan the 1st?
#     #     T, m, n = img2.shape # we need some time T!. 
#     #     img1_pad = np.pad(img1,[[0,0], [m-1,m-1],[n-1,n-1]], mode='edge') # do the padding
#     #     # img1_pad = np.pad(img1,[[0,0], [m-1,m-1],[n-1,n-1]], mode='reflect')#, constant_values=0)
#     #     out = np.zeros((m+n-1, m+n-1)) # full # granger causal intensities!. 
#     #     # out2 = np.zeros((m+n-1, m+n-1)) # full
#     #     """
#     #     we can speed this up by extracting all windows... and running list comprehensin ? 
#     #     """
#     #     M,N = out.shape
#     #     # for ii in tqdm(np.arange(M)):
#     #     for ii in np.arange(0,M):
#     #         for jj in np.arange(0,N):
#     #             Y = img1_pad[:,ii:ii+m,
#     #                            jj:jj+n].copy()
#     #             X = img2.copy()
#     #             # Y = Y.reshape(Y.shape[0], -1)
#     #             # X = X.reshape(X.shape[0], -1)
#     #             Y = Y.reshape(-1, Y.shape[-1]).T
#     #             X = X.reshape(-1, X.shape[-1]).T
#     #             X = X - X.mean()
#     #             X = Y - Y.mean()
#     #             calc_xy = CausalCalculator(X=Y, Y_cause=X)
#     #             Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1) # delay lag=1 and order=1 # this is slow.... 
#     #             out[ii,jj] = Gy_to_x # scalar. 
#     #     return out
    
#     def granger_naive(img1, img2):
#         from CausalCalculator import CausalCalculator
#         ## zero_pad # we scan the 1st?
#         m, n, T = img2.shape # we need some time T!. 
#         img1_pad = np.pad(img1,[[m-1,m-1],[n-1,n-1], [0,0]], mode='constant', constant_values=0) # do the padding
#         # img1_pad = np.pad(img1,[[m-1,m-1],[n-1,n-1], [0,0]], mode='edge') # do the padding
#         # img1_pad = np.pad(img1,[[0,0], [m-1,m-1],[n-1,n-1]], mode='reflect')#, constant_values=0)
#         out = np.zeros((m+n-1, m+n-1)) # full # granger causal intensities!. 
#         # out2 = np.zeros((m+n-1, m+n-1)) # full
#         """
#         we can speed this up by extracting all windows... and running list comprehensin ? 
#         """
#         M,N = out.shape
#         # for ii in tqdm(np.arange(M)):
#         for ii in np.arange(0,M):
#             for jj in np.arange(0,N):
#                 Y = img1_pad[ii:ii+m,
#                              jj:jj+n,:].copy()
#                 X = img2.copy()
#                 # print(X.shape, Y.shape)
#                 # Y = Y.reshape(Y.shape[0], -1)
#                 # X = X.reshape(X.shape[0], -1)
#                 Y = Y.reshape(-1, Y.shape[-1]).T
#                 X = X.reshape(-1, X.shape[-1]).T
#                 # print(X.shape, Y.shape)
#                 X = X - X.mean()
#                 X = Y - Y.mean()
                
#                 # its like on or off.... 
#                 calc_xy = CausalCalculator(X=X, Y_cause=Y) # for some reason .... no magnitude.... 
#                 Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1) # delay lag=1 and order=1 # this is slow.... 
#                 out[ii,jj] = Gy_to_x # scalar. 
                
#         # print(out.shape)
#         return out 
#         # return out[out.shape[0]//2-m//2:out.shape[0]//2-m//2+m, out.shape[1]//2-n//2:out.shape[1]//2-n//2+n]
    
#     def granger_naive2(img1, img2):
#         from CausalCalculator import CausalCalculator
#         ## zero_pad # we scan the 1st?
#         m1, n1, T1 = img1.shape
#         m2, n2, T2 = img2.shape # we need some time T!. 
        
#         shifts_m = m2-m1+1
#         shifts_n = n2-n1+1
        
#         out = np.zeros((shifts_m, shifts_n)) # full # granger causal intensities!. 
#         # out2 = np.zeros((m+n-1, m+n-1)) # full
#         """
#         we can speed this up by extracting all windows... and running list comprehensin ? 
#         """
#         M,N = out.shape
#         # print(M,N)
#         # for ii in tqdm(np.arange(M)):
#         for ii in np.arange(0,M):
#             for jj in np.arange(0,N):
#                 Y = img2[ii:ii+m1,
#                          jj:jj+n1,:].copy()
#                 X = img1.copy()
#                 # print(X.shape, Y.shape)
#                 # Y = Y.reshape(Y.shape[0], -1)
#                 # X = X.reshape(X.shape[0], -1)
#                 Y = Y.reshape(-1, Y.shape[-1]).T
#                 X = X.reshape(-1, X.shape[-1]).T
#                 X = X - X.mean()
#                 X = Y - Y.mean()
#                 calc_xy = CausalCalculator(X=X, Y_cause=Y)
#                 Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1) # delay lag=1 and order=1 # this is slow.... 
#                 out[ii,jj] = Gy_to_x # scalar. 
#         return out
    
#     def pcca_cause(img1, img2):
        
#         from CausalCalculator import CausalCalculator
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         Y = Y.reshape(-1, Y.shape[-1]).T
#         X = X.reshape(-1, X.shape[-1]).T
#         # print(X.shape, Y.shape)
#         # X = X - X.mean(axis=1)[:,None]
#         # X = Y - Y.mean(axis=1)[:,None]
#         # X = (X - X.mean())/(X.std()+1e-8) 
#         # Y = (Y - Y.mean())/(Y.std()+1e-8) 
#         # print(X.shape)
#         # print(Y.shape)
#         calc_xy = CausalCalculator(X=X, Y_cause=Y)
#         # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
#         #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
#         Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=3,
#                                                eta_xt=5e-4, 
#                                                eta_yt=5e-4, 
#                                                eta_xtkm=5e-4) # etas are very important 
#         return Gy_to_x
    
    
#     def TE_cause(img1, img2):
        
#         import te_compute as te
#         from copent import transent
        
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         # xs = np.mean()
#         # myTE = te.te_compute(xs, ys, 1, 1)
        
#         Y = Y.reshape(-1, Y.shape[-1]).T
#         X = X.reshape(-1, X.shape[-1]).T
        
#         # print(X.shape, Y.shape)
#         xs = np.nanmean(X, axis=-1)
#         ys = np.nanmean(Y, axis=-1)
#         # print(xs.shape, ys.shape)
        
#         # # is the order wrong? 
#         TE_1 = te.te_compute(ys, #-np.nanmean(ys), 
#                             xs,#-np.nanmean(xs), 
#                             5, 3) # number of nearest neighbors needs to increase
#         TE_2 = te.te_compute(xs, #-np.nanmean(ys), 
#                             ys,#-np.nanmean(xs), 
#                             5, 3) # number of nearest neighbors needs to increase
        
#         TE = np.maximum(TE_1,TE_2) - np.minimum(TE_1,TE_2)
#         # TE = transent(ys, xs, lag = 5, k = 3, dtype = 2, mode = 1)
#         # # print(X.shape, Y.shape)
#         # # X = X - X.mean(axis=1)[:,None]
#         # # X = Y - Y.mean(axis=1)[:,None]
#         # # X = (X - X.mean())/(X.std()+1e-8) 
#         # # Y = (Y - Y.mean())/(Y.std()+1e-8) 
#         # # print(X.shape)
#         # # print(Y.shape)
#         # calc_xy = CausalCalculator(X=X, Y_cause=Y)
#         # # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
#         # #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
#         # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=3,
#         #                                        eta_xt=5e-4, 
#         #                                        eta_yt=5e-4, 
#         #                                        eta_xtkm=5e-4) # etas are very important 
#         return TE
    
    
#     def PDF_cause_hack(img1, img2, p=3):
        
#         from pdc_dtf import mvar_fit, PDC
        
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         # xs = np.mean()
#         # myTE = te.te_compute(xs, ys, 1, 1)
        
#         Y = Y.reshape(-1, Y.shape[-1])
#         X = X.reshape(-1, X.shape[-1])
#         N = len(Y)
#         # compute all the causalities. 
        
#         Y_ = np.vstack([X,Y])
#         mu = np.mean(Y_, axis=1)
#         X_ = Y_ - mu[:, None]
        
#         A_est, sigma = mvar_fit(X_, p)    
#         sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
#         # sigma = None)
    
#         # compute PDC
#         # print(A_est.shape)
#         P, freqs = PDC(A_est, sigma)
        
#         # get the causalities of the block between X to Y.... 
#         P_xy = P[:, -N:, :N].copy() # this should be X -> Y 
#         # print(P_xy.shape)
#         P_xy = np.max(P_xy, axis=0) # maximum over all frequencies.... 
#         # print(P_xy.shape)
#         # we need to subtract this from some base... 
#         # return np.log(np.linalg.det(P_xy))
#         return np.prod(np.linalg.slogdet(P_xy))
    
    
#     def TE_pyinform_cause(img1, img2):
        
#         # import te_compute as te
#         from pyinform.transferentropy import transfer_entropy
        
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         # xs = np.mean()
#         # myTE = te.te_compute(xs, ys, 1, 1)
        
#         Y = Y.reshape(-1, Y.shape[-1]).T
#         X = X.reshape(-1, X.shape[-1]).T
        
#         # # print(X.shape, Y.shape)
#         xs = np.nanmean(X, axis=-1)
#         ys = np.nanmean(Y, axis=-1)
        
#         # xs = np.nanmax(X, axis=-1)
#         # ys = np.nanmax(Y, axis=-1)
#         # print(xs.shape, ys.shape)
        
#         # is the order wrong? 
#         TE = transfer_entropy(ys, 
#                               xs, k=1) # number of nearest neighbors needs to increase
#         # # print(X.shape, Y.shape)
#         # # X = X - X.mean(axis=1)[:,None]
#         # # X = Y - Y.mean(axis=1)[:,None]
#         # # X = (X - X.mean())/(X.std()+1e-8) 
#         # # Y = (Y - Y.mean())/(Y.std()+1e-8) 
#         # # print(X.shape)
#         # # print(Y.shape)
#         # calc_xy = CausalCalculator(X=X, Y_cause=Y)
#         # # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
#         # #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
#         # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=3,
#         #                                        eta_xt=5e-4, 
#         #                                        eta_yt=5e-4, 
#         #                                        eta_xtkm=5e-4) # etas are very important 
#         return TE
    
    
#     def differential_covariance(X, eps=1e-12, reg=1e-3):
    
#         import numpy as np 
        
#         # standardize
#         X_ = (X - np.nanmean(X, axis=1)[:,None]) #/ ( np.nanstd(X, axis=1)[:,None] + eps )
#         # X_ = X.copy()
        
#         # # differential 
#         # X_pad = np.pad(X_, pad_width=[[0,0],[1,1]], mode='edge')
#         # dX_ = (X_pad[:,2:] - X_pad[:,:-2]) / 2.
#         # # dX_ = np.gradient(X_, axis=1)
#         dX_ = X_[:,1:] - X_[:,:-1]
#         X_ = X_[:,1:].copy()
#         # print(X_.shape,dX_.shape)
#         X_ = X_.T
#         dX_ = dX_.T
        
#         # linear least squares solution . 
#         dX_X = dX_.T.dot(X_)
#         X_X = X_.T.dot(X_)
        
#         # W = np.linalg.solve(X_X, dX_X) # transpose... 
#         W = dX_X.dot(np.linalg.inv(X_X+reg*np.eye(len(X_X))))
        
#         return W 
    
    
#     def GC_full_reduced_separate_regress(img1, img2, lag=1, alpha=.1):
        
#         from sklearn.linear_model import Ridge
#         import numpy as np 
            
#         # initialise 
        
#         """
#         Reduced regression 
#         """
#         # reduced model 
#         clf = Ridge(alpha=alpha)

#         Y = (img1.reshape(-1,centre_ref.shape[-1]).T)[lag:]
#         X = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X.append(X_)
#         X = np.hstack(X)
        
#         clf.fit(X,Y)
        
#         # logL = np.prod(np.linalg.slogdet(np.cov(Y - clf.predict(X))))
#         logL = np.log(np.var(Y - clf.predict(X), axis=0).mean())
        
#         """
#         Full Regression
#         """
#         # full model 
#         clf_full = Ridge(alpha=alpha)
#         X_full = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         for ll in range(1,lag+1):
#             X_ = (img2.reshape(-1,img2.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         X_full.append((img2.reshape(-1,img2.shape[-1]).T)[lag:])
#         X_full = np.hstack(X_full) # n_time x n_variables.
        
#         clf_full.fit(X_full, Y)
        
#         # logF = np.prod(np.linalg.slogdet(np.cov(Y - clf_full.predict(X_full))))
#         logF = np.log(np.var(Y - clf_full.predict(X_full), axis=0).mean())
        
        
#         # get the difference!. # not a pval ... but a magnitude. 
#         return logF - logL 
    
    
#     def GC_full_reduced_separate_regress_individual(img1, img2, lag=1, alpha=.1):
        
#         from sklearn.linear_model import Ridge
#         import numpy as np 
            
#         # initialise 
#         """
#         Reduced regression 
#         """
#         # reduced model 
#         clf = Ridge(alpha=alpha)

#         Y = (img1.reshape(-1,centre_ref.shape[-1]).T)[lag:]
#         X = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X.append(X_)
#         X = np.hstack(X)
        
#         clf.fit(X,Y)
        
#         # logL = np.prod(np.linalg.slogdet(np.cov(Y - clf.predict(X))))
#         logL = np.log(np.var(Y - clf.predict(X), axis=0)) # .mean())
        
#         """
#         Full Regression
#         """
#         # full model 
#         clf_full = Ridge(alpha=alpha)
#         X_full = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         for ll in range(1,lag+1):
#             X_ = (img2.reshape(-1,img2.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         X_full.append((img2.reshape(-1,img2.shape[-1]).T)[lag:])
#         X_full = np.hstack(X_full) # n_time x n_variables.
        
#         clf_full.fit(X_full, Y)
        
#         # logF = np.prod(np.linalg.slogdet(np.cov(Y - clf_full.predict(X_full))))
#         logF = np.log(np.var(Y - clf_full.predict(X_full), axis=0)) #.mean())
        
        
#         # get the difference!. # not a pval ... but a magnitude. 
#         return logF - logL 
        

#     import scipy.stats as spstats
#     from pdc_dtf import mvar_fit, PDC
#     row_ii = 0
#     row_jj = 0
    
#     for row_ii in tqdm(np.arange(len(row_indices[:]))):
#         # for row_jj in tqdm(np.arange(len(col_indices[:]))): 
#         for row_jj in np.arange(len(col_indices[:])): 
#             """
#             need to evaluate an actual convolution.... 
#             """
            
#             # starting coordinates
#             rr = row_indices[row_ii]#-winsize//2
#             cc = col_indices[row_jj]#-winsize//2
            
#             xy_coords[row_ii, row_jj,0] = cc - winsize//2
#             xy_coords[row_ii, row_jj,1] = rr - winsize//2
            
#             corr_array = np.zeros((3,3))
#             # # corr_array_vects = np.dstack([[[-1,-1,-1],
#             # #                               [0,0,0],
#             # #                               [1,1,1]], [[-1,0,1],
#             # #                                           [-1,0,1],
#             # #                                           [-1,0,1]]])
#             # # corr_array_vects = corr_array_vects/(np.linalg.norm(corr_array_vects, axis=-1)[...,None]+1e-8)
            
#             # last frame 
#             centre_ref = frame_a_[rr:rr+winsize,cc:cc+winsize].copy() # the last axis is time!.
            
#             # # next frame 
#             top_left = frame_b_[rr-winsize:rr,cc-winsize:cc].copy()
#             top_center = frame_b_[rr-winsize:rr,cc:cc+winsize].copy()
#             top_right = frame_b_[rr-winsize:rr,cc+winsize:cc+2*winsize].copy()
    
#             # # # corr_array[0,0] = spstats.pearsonr(centre_ref.ravel(), top_left.ravel())[0]
#             # # # corr_array[0,1] = spstats.pearsonr(centre_ref.ravel(), top_center.ravel())[0]
#             # # # corr_array[0,2] = spstats.pearsonr(centre_ref.ravel(), top_right.ravel())[0]
#             # # corr_array[0,0] = GC_full_reduced_separate_regress(top_left, centre_ref, lag=5, alpha=.1) #(centre_ref, top_left)
#             # # corr_array[0,1] = GC_full_reduced_separate_regress(top_center, centre_ref, lag=5, alpha=.1)#(centre_ref, top_center)
#             # # corr_array[0,2] = GC_full_reduced_separate_regress(top_right, centre_ref, lag=5, alpha=.1)#(centre_ref, top_right)
#             # corr_array[0,0] = GC_full_reduced_separate_regress(centre_ref, top_left, lag=5, alpha=.1) #(centre_ref, top_left)
#             # corr_array[0,1] = GC_full_reduced_separate_regress(centre_ref, top_center, lag=5, alpha=.1)#(centre_ref, top_center)
#             # corr_array[0,2] = GC_full_reduced_separate_regress(centre_ref, top_right, lag=5, alpha=.1)#(centre_ref, top_right)
    
#             left = frame_b_[rr:rr+winsize,cc-winsize:cc].copy()
#             center = frame_b_[rr:rr+winsize,cc:cc+winsize].copy()
#             # center = frame_b_[rr-winsize//2:rr+winsize+winsize//2,cc-winsize//2:cc+winsize+winsize//2].copy()
#             # center = frame_b_[rr-winsize:rr+winsize+winsize,cc-winsize:cc+winsize+winsize].copy()
#             right = frame_b_[rr:rr+winsize,cc+winsize:cc+2*winsize].copy()
            
#             # # # corr_array[1,0] = spstats.pearsonr(centre_ref.ravel(), left.ravel())[0]
#             # # # corr_array[1,1] = spstats.pearsonr(centre_ref.ravel(), center.ravel())[0]
#             # # # corr_array[1,2] = spstats.pearsonr(centre_ref.ravel(), right.ravel())[0]
#             # # corr_array[1,0] = GC_full_reduced_separate_regress(left, centre_ref, lag=5, alpha=.1) #(centre_ref, left)
#             # # corr_array[1,1] = GC_full_reduced_separate_regress(center, centre_ref, lag=5, alpha=.1) #(centre_ref, center)
#             # # corr_array[1,2] = GC_full_reduced_separate_regress(right, centre_ref, lag=5, alpha=.1) #(centre_ref, right)
#             # corr_array[1,0] = GC_full_reduced_separate_regress(centre_ref, left, lag=5, alpha=.1) #(centre_ref, left)
#             # corr_array[1,1] = GC_full_reduced_separate_regress(centre_ref, center, lag=5, alpha=.1) #(centre_ref, center)
#             # corr_array[1,2] = GC_full_reduced_separate_regress(centre_ref, right, lag=5, alpha=.1) #(centre_ref, right)
            
#             bottom_left = frame_b_[rr+winsize:rr+2*winsize, cc-winsize:cc].copy()
#             bottom_center = frame_b_[rr+winsize:rr+2*winsize, cc:cc+winsize].copy()
#             bottom_right = frame_b_[rr+winsize:rr+2*winsize, cc+winsize:cc+2*winsize].copy()
            
#             # # # corr_array[2,0] = spstats.pearsonr(centre_ref.ravel(), bottom_left.ravel())[0]
#             # # # corr_array[2,1] = spstats.pearsonr(centre_ref.ravel(), bottom_center.ravel())[0]
#             # # # corr_array[2,2] = spstats.pearsonr(centre_ref.ravel(), bottom_right.ravel())[0]
#             # # corr_array[2,0] = GC_full_reduced_separate_regress(bottom_left, centre_ref, lag=5, alpha=.1) #(centre_ref, bottom_left)
#             # # corr_array[2,1] = GC_full_reduced_separate_regress(bottom_center, centre_ref, lag=5, alpha=.1)#(centre_ref, bottom_center)
#             # # corr_array[2,2] = GC_full_reduced_separate_regress(bottom_right, centre_ref, lag=5, alpha=.1)#(centre_ref, bottom_right)
#             # corr_array[2,0] = GC_full_reduced_separate_regress(centre_ref, bottom_left, lag=5, alpha=.1) #(centre_ref, bottom_left)
#             # corr_array[2,1] = GC_full_reduced_separate_regress(centre_ref, bottom_center, lag=5, alpha=.1)#(centre_ref, bottom_center)
#             # corr_array[2,2] = GC_full_reduced_separate_regress(centre_ref, bottom_right, lag=5, alpha=.1)#(centre_ref, bottom_right)
    
#             # corr_array[np.isnan(corr_array)] = 0 
#             # corr_array[1,1] = 0 #np.nanmean(corr_array[corr_array>0])
#             # corr_array = normxcorr2(centre_ref, center)
            
#             """
#             Generate the block correlation .... with sliding windows... 
#             """
            
#             # corr_array = GC_full_reduced_separate_regress_individual(centre_ref-centre_ref.mean(axis=-1)[...,None], 
#             #                                                          center-center.mean(axis=-1)[...,None], 
#             #                                                          lag=5, alpha=1) #(centre_ref, center)
#             corr_array = GC_full_reduced_separate_regress_individual(centre_ref, 
#                                                                      center, 
#                                                                      lag=5, alpha=1) #(centre_ref, center)
#             corr_array = corr_array.reshape((winsize,winsize))
#             # # # corr_array = granger_naive(centre_ref, center) # this scan needs to be completely reconfigured.... to accomodate for different sliding... 
#             # # # # corr_array = granger_naive2(centre_ref, center)
#             # # # # # corr_array[corr_array==0] = np.nan
#             # corr_array[corr_array>0] = 0
#             # corr_array = np.abs(corr_array)
            
#             # plt.figure()
#             # plt.imshow(corr_array)
#             # plt.show()
            
#             # # ### For PCCA maybe this below is best - check this reproduces... (if does we expand out of the block)
#             # # # corr_array_grad = np.array(np.gradient(np.abs(corr_array)))
#             # # corr_array_grad = np.array(np.gradient(corr_array)) # this finds homogeneity!!!,,, i see... 
#             # # mean_vector = np.nansum(corr_array_grad.reshape(2,-1), axis=-1) # this should be correct now. 
#             # # mean_vector = np.sum(corr_array) * mean_vector # add back the intensity... 
#             # YY, XX = np.indices(corr_array.shape)
#             # YY_ = YY - corr_array.shape[0]//2
#             # XX_ = XX - corr_array.shape[1]//2
            
#             # corr_array_vects = np.dstack([YY_,XX_])
#             # corr_array_vects = corr_array_vects/(np.linalg.norm(corr_array_vects, axis=-1)[...,None] + 1e-8)
            
#             # # # corr_vectors = corr_array_vects*np.clip(corr_array[...,None],0,1)
#             # corr_vectors = corr_array_vects*corr_array[...,None] 
#             # mean_vector = np.nanmean(corr_vectors.reshape(-1,2), axis=0) * np.sum(corr_array) 
            
#             mid = corr_array.shape[1]//2
            
#             corr_x_direction = -corr_array[:,:mid].sum() + corr_array[:,mid+1:].sum()
#             corr_y_direction = -corr_array[:mid].sum() + corr_array[mid+1:].sum()
#             intensity = np.sum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
            
#             mean_vector = np.hstack([corr_y_direction, corr_x_direction])
#             mean_vector = mean_vector * intensity
#             # # # # # # simply the total strength of stuff flowing past the central pixel 
            
#             # # # """ this should still be correct for directionality ... """
#             # # mean_vector = np.hstack([np.nansum(corr_array[:,corr_array.shape[0]//2+1:]) - np.nansum(corr_array[:,:corr_array.shape[0]//2]),
#             # #                          np.nansum(corr_array[corr_array.shape[0]//2+1:]) - np.nansum(corr_array[:corr_array.shape[0]//2])])
#             # # mean_vector = np.sum(corr_array) * mean_vector
#             # # mean_vector = mean_vector[::-1]
#             # # # # out_vect[row_ii,row_jj,:] = mean_vector[::-1]
#             out_vect[row_ii,row_jj,:] = -mean_vector
            
#             row_jj += 1
#         row_ii +=1
            
#     import scipy.ndimage as ndimage
    
#     # # now we  apply smoothing to derive the flows!. ---> this is quite important!. 
#     out_vect[...,0] = ndimage.gaussian_filter(out_vect[...,0], sigma=1)
#     out_vect[...,1] = ndimage.gaussian_filter(out_vect[...,1], sigma=1)
    
#     # plot the vector field
#     sampling = 1
    
#     plt.figure(figsize=(15,15))
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 out_vect[::sampling,::sampling,1],  # x 
#                 -out_vect[::sampling,::sampling,0]) # y 
#     plt.show()
    

#     # plt.figure(figsize=(15,15))
#     # # plt.imshow(myVid[1])
#     # plt.quiver(XX, 
#     #            YY, 
#     #             corr_array_vects[...,1], 
#     #             corr_array_vects[...,0])
#     # plt.show()


# # # #myVidGrey = np.zeros([myVid.shape[0], myVid.shape[1], myVid.shape[2]])
# # # myVidGrey = np.zeros([194, 60, 90 ])
# # # for i in range(0, myVid.shape[0]):
# # #     myGrey = color.rgb2gray(myVid[i,:,:,])
# # #     myGreyRescale = rescale(myGrey, .125)
# # #     myVidGrey[i,:,:] = myGreyRescale


# """
# This is correct!..... 
# """
# optical_flow_params = dict(pyr_scale=0.5, levels=1, winsize=5, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
 
# vid_flow = extract_optflow(255*myVid[start:end][:], 
#                             optical_flow_params, 
#                             rescale_intensity=False, 
#                             intensity_range=[2,98])

# YY, XX = np.indices(myVid.shape[1:3])
# sampling = 3 

# plt.figure(figsize=(15,15))
# plt.subplot(121)
# plt.imshow(myVid[1])
# # plt.quiver(xy_coords[:,:,0], 
# #             xy_coords[:,:,1], 
# #             out_vect[:,:,1], 
# #             -out_vect[:,:,0])
# plt.quiver(xy_coords[:,:,0], 
#             xy_coords[:,:,1], 
#             out_vect[:,:,1], 
#             -out_vect[:,:,0])
# plt.subplot(122)
# plt.imshow(myVid[1,:,:])
# # plt.quiver(XX[::sampling,::sampling], 
# #             YY[::sampling,::sampling], 
# #             np.nanmean(vid_flow[:20], axis=0)[::sampling,::sampling,0], 
# #             -np.nanmean(vid_flow[:20],axis=0)[::sampling,::sampling,1])
# plt.quiver(XX[::sampling,::sampling], 
#             YY[::sampling,::sampling], 
#             vid_flow.mean(axis=0)[::sampling,::sampling,0], 
#             -vid_flow.mean(axis=0)[::sampling,::sampling,1])
# plt.show()






    
# plt.imshow(myVidGrey[1,:,:])
# plt.title('Example Snapshot of GreScale Video')

# myMat = myVidGrey
# transferEntropyVector = np.zeros([myMat.shape[1], myMat.shape[2], 2])

# myK = 1
# thresh = 20

# from tqdm import tqdm 

# ##for each  valid box in matCellRatioT compute transfer entropy with lag 1
# for i in tqdm(range(1,myMat.shape[1] -1)):
#     # print(i)
#     for j in range(1,myMat.shape[2] -1):
#         # print(j)
#         if (sum(myMat[:,i,j] > 0) > thresh):
#             xs = myMat[1:20, i,j]
#             xs = (xs - xs.mean()) / xs.std()
            
#             if  (sum(myMat[:, i+1,j] > 0) > thresh):
#                 ys = myMat[2:21, (i+1), j]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([1,0]) * transfer_entropy(xs, ys, k= 1)
#                 newVal = np.array([1,0]) * te.te_compute(xs, ys,  1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal
#             if  (sum(myMat[:, i-1,j] > 0) > thresh):
#                 ys = myMat[2:21, (i-1), j]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([-1,0]) * transfer_entropy(xs, ys, k = 1)
#                 newVal = np.array([-1,0]) * te.te_compute(xs, ys, 1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal
#             if  (sum(myMat[:, i,j+1] > 0) > thresh):
#                 ys = myMat[2:21, i, (j+1)]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([0,1]) * transfer_entropy(xs, ys, k= 1)
#                 newVal = np.array([0,1]) * te.te_compute(xs, ys, 1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal   
#             if  (sum(myMat[:, i,j-1] > 0) > thresh):
#                 ys = myMat[2:21, i, (j-1)]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([0,-1]) * transfer_entropy(xs, ys, k=1)
#                 newVal = np.array([0,-1]) * te.te_compute(xs, ys, 1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal   

# plt.imshow(myMat[20,:,:])
# plt.title('Rac: Frame from Ratiometric Movie')


# YY, XX = np.indices(myVidGrey.shape[1:3])

# plt.figure()
# plt.imshow(myMat[1,:,:])
# plt.quiver(XX, 
#            YY, 
#            transferEntropyVector[:,:,0], 
#            -transferEntropyVector[:,:,1], scale=50)
# plt.show()

# ######
# transferEntropyVector[np.isnan(transferEntropyVector)] = 0               
# # compute the cumulative gradients of transfer entropy gradient
# gradx = np.gradient(transferEntropyVector[:,:,0])
# gradxSum = np.array(abs(gradx[0]) + abs(gradx[1]))
# grady = np.gradient(transferEntropyVector[:,:,1])
# gradySum = np.array(abs(grady[0]) + abs(grady[1]))
# totalGradient = (gradxSum + gradySum)
# plt.imshow(totalGradient)
# plt.title('Sum of abs (xGradient), abs(yGradient)')
# #clean up weird grad
# totalGradient[totalGradient >50] = 0

# #totalGradient[totalGradient < 10] = 0
# #Use difference of Gaussian and treshholding to determine microdomains
# differenceOfGaussian = gaussian_filter(totalGradient, 1) - gaussian_filter(totalGradient, 20)
# differenceOfGaussian[differenceOfGaussian < (np.mean(differenceOfGaussian) + 3 * np.std(differenceOfGaussian))] = 0
# plt.imshow(differenceOfGaussian, interpolation = 'nearest')
# plt.title('Rac')

# plt.show()

# #plt.imshow(differenceOfGaussian, interpolation = 'nearest')
# #plt.show()
# #########################################################
# plt.imshow(abs(transferEntropyVector[:,:,1]) + abs(transferEntropyVector[:,:,0]))
# plt.title('Sum of absolute value of TE vector in both X and Y direction, Rac Cell1')


