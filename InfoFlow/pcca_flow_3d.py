import numpy as np
from InfoFlow.pcca_flow import pcca_cause


def pcca_cause_block_3d(img1, img2, block_size_xy=3, block_size_z=3, **kwargs):
    """
    img1, img2: (3*bsz, 3*bsxy, 3*bsxy, T)
    Returns: (3, 3, 3) causal scores over 3x3x3 neighbourhood
    """
    bsz, bsxy = block_size_z, block_size_xy
    corr_array = np.zeros((3, 3, 3))
    centre_ref = img1[bsz:2*bsz, bsxy:2*bsxy, bsxy:2*bsxy]
    for iz in range(3):
        for iy in range(3):
            for ix in range(3):
                neighbour = img2[iz*bsz:(iz+1)*bsz,
                                 iy*bsxy:(iy+1)*bsxy,
                                 ix*bsxy:(ix+1)*bsxy]
                corr_array[iz, iy, ix] = pcca_cause(centre_ref, neighbour, **kwargs)
    return corr_array
