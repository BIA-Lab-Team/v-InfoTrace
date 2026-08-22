"""
Example script illustrating the 3D (volumetric) InfoFlow drivers in
InfoFlow/infoflow3D.py, in the same spirit as
2023-03-31_testScript_InfoFlow_shorter.py (run each causal measure, visualize
the resulting flow) -- but for a (T,Z,H,W) volume instead of a (T,H,W) video.

There's no ready volumetric dataset in this repo, so a synthetic drifting
Gaussian blob is used instead (same construction as test_infoflow3d.py's
ground-truth check), which also makes the expected flow direction obvious
for a sanity check by eye. Since a single 2D quiver plot can't show a
3-component (z,y,x) flow field directly, this visualizes the (y,x) components
at the blob's mid-depth Z-slice and reports the z-component separately as a
per-depth mean profile.

Usage:
    python 2026-08-22_exampleScript_InfoFlow_3D.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from InfoFlow.infoflow3D import (
    causal_flow_scores_3d,
    causal_block_flow_3d,
    causal_block_flow_scores_gradient_3d,
)
from InfoFlow.DDC_flow_3d import DDC_cause_3d
from InfoFlow.LK_flow_3d import Linear_LK_cause_3d
from InfoFlow.pdc_dtf_flow_3d import PDC_central_flow_3d
from InfoFlow.gc_flow_3d import GC_full_reduced_separate_regress_individual_3d
from InfoFlow.correlation_flow_3d import nd_xcorr_lag_3d
from InfoFlow.pcca_flow_3d import pcca_cause_block_3d
from InfoFlow.flow_vis import flow_to_color

OUTDIR = "results/example_3d"
os.makedirs(OUTDIR, exist_ok=True)

"""
1. build a synthetic (T,Z,H,W) volume: a Gaussian blob drifting at a known
   velocity (vz,vy,vx) voxels/frame, standing in for a real volumetric video.
"""
RNG = np.random.default_rng(0)
T, Z, H, W = 30, 12, 24, 24
vz, vy, vx = 0.0, 0.4, 0.3   # drift diagonally in y and x, static in z

zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(H), np.arange(W), indexing='ij')
z0, y0, x0 = Z // 2, H // 4, W // 4

myVol = np.zeros((T, Z, H, W))
for t in range(T):
    r2 = (((zz - (z0 + t*vz)) / 2)**2
        + ((yy - (y0 + t*vy)) / 2)**2
        + ((xx - (x0 + t*vx)) / 2)**2)
    myVol[t] = np.exp(-r2)
myVol = myVol + 0.05 * RNG.standard_normal(myVol.shape)

print(f"Volume shape: {myVol.shape}  (T,Z,H,W)")
print(f"True drift  : vz={vz}, vy={vy}, vx={vx}  (voxels/frame)")

mid_t = T // 2
mid_z = int(round(z0 + mid_t * vz))
print(f"Mid-depth slice for visualization: z={mid_z}")

# =============================================================================
#     2. run through all point-wise causal measures and extract the output
# =============================================================================
WZ, WXY = 3, 3  # cubic windows, matches PCCA's block-design requirement below

causal_flow_outputs_3d = [
    causal_flow_scores_3d(myVol, DDC_cause_3d, winsize_xy=WXY, winsize_z=WZ, eps=1e-12, alpha=1e-2),
    causal_flow_scores_3d(myVol, Linear_LK_cause_3d, winsize_xy=WXY, winsize_z=WZ, eps=1e-12, alpha=1e-2),
    causal_flow_scores_3d(myVol, GC_full_reduced_separate_regress_individual_3d, winsize_xy=WXY, winsize_z=WZ, lag=1, alpha=1),
    causal_flow_scores_3d(myVol, nd_xcorr_lag_3d, winsize_xy=WXY, winsize_z=WZ, lag=1),
]
causal_flow_vectors_3d = [causal_block_flow_scores_gradient_3d(flo) for flo in causal_flow_outputs_3d]

# PCCA uses the block driver directly (its own internal neighbor decomposition), as in 2D.
pcca_flow_3d = causal_block_flow_3d(myVol, pcca_cause_block_3d, winsize_xy=WXY, winsize_z=WZ,
                                     k=1, m=1, eta_xt=5e-4, eta_yt=5e-4, eta_xtkm=5e-4)

flow_methods_3d = ['DDC', 'LK', 'GC', 'Corr']
all_flows_3d = causal_flow_vectors_3d + [pcca_flow_3d]
flow_methods_3d = flow_methods_3d + ['PCCA']

# =============================================================================
#     3. Visualization: (y,x) quiver + flow-color at the blob's mid-depth
#        slice, plus a per-depth mean z-flow profile.
# =============================================================================
xy_coords = np.indices(myVol.shape[2:]); xy_coords = xy_coords.transpose(1, 2, 0)
xy_coords = xy_coords[..., ::-1]

bg = myVol[mid_t, mid_z]
sampling = 1

for name, flow in zip(flow_methods_3d, all_flows_3d):
    # flow shape: (Z,H,W,3) with last axis = (z,y,x)
    flow_slice = flow[mid_z]  # (H,W,3)
    flow_yx = flow_slice[..., 1:]  # (H,W,2) = (y,x), matches the 2D quiver convention

    plt.figure(figsize=(8, 8))
    plt.title(f"{name} 3D flow (y,x) at z={mid_z}")
    plt.imshow(bg, cmap='gray')
    plt.quiver(xy_coords[::sampling, ::sampling, 0],
               xy_coords[::sampling, ::sampling, 1],
               flow_yx[::sampling, ::sampling, 1],   # x
               -flow_yx[::sampling, ::sampling, 0])  # y
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{name}_3d_quiver_yx.png"), dpi=130)
    plt.close()

    flow_color = flow_to_color(flow_yx[..., ::-1])
    plt.figure(figsize=(8, 8))
    plt.title(f"{name} 3D flow-color (y,x) at z={mid_z}")
    plt.imshow(bg, cmap='gray')
    plt.imshow(flow_color, alpha=0.7)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{name}_3d_flowcolor_yx.png"), dpi=130)
    plt.close()

    z_profile = np.nanmean(flow[..., 0], axis=(1, 2))  # mean z-flow per depth plane
    plt.figure(figsize=(6, 4))
    plt.title(f"{name}: mean z-flow per depth plane (true vz={vz})")
    plt.plot(z_profile, marker='o')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.xlabel('Z plane')
    plt.ylabel('mean flow_z')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{name}_3d_zprofile.png"), dpi=130)
    plt.close()

    mean_yx = np.nanmean(flow_slice[..., 1:], axis=(0, 1))
    print(f"  {name}: mean(y,x) at z={mid_z} = ({mean_yx[0]:.4f}, {mean_yx[1]:.4f})  "
          f"(true vy={vy}, vx={vx}) -> saved {name}_3d_*.png")

print(f"\nAll figures saved under {OUTDIR}/")
