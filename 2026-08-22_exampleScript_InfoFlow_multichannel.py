"""
Example script illustrating the multi-channel ("joint") InfoFlow drivers,
in the same spirit as 2023-03-31_testScript_InfoFlow_shorter.py (load video,
run each causal measure, visualize as quiver plots + flow-color overlays) --
but using all raw channels of a video simultaneously instead of converting
to grayscale, and reading off both within-channel and cross-channel flow.

Unlike the single-channel demo, each measure here returns a dict keyed by
(source_channel, target_channel), one flow field per ordered pair. This
script shows two representative pairs per measure: the within-channel flow
of channel 0, and the cross-channel flow from channel 0 into channel 1.

Usage:
    python 2026-08-22_exampleScript_InfoFlow_multichannel.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless; figures are saved to disk instead of plt.show()
import pylab as plt

"""
Imports of the multichannel joint driver functions.
"""
import InfoFlow.infoflow as infoflow_scripts
from InfoFlow.pcca_flow import pcca_cause_confound_native
from InfoFlow.flow_vis import flow_to_color

OUTDIR = "results/example_multichannel"
os.makedirs(OUTDIR, exist_ok=True)

"""
1. read in the video file, keeping all 3 raw (B,G,R) channels -- no
   rgb2gray conversion, since the whole point here is to treat the channels
   as separate signals rather than collapsing them.
"""
myVid_rgb = infoflow_scripts.read_video_cv2(r'data/UCF_CrowdsDataset/3687-18_70.mov')
myVid_rgb = myVid_rgb[:40].astype(np.float64)  # keep it short for a quick example run
channel_names = ['B', 'G', 'R']

"""
downscale each channel for speed and smooth at gaussian sigma 1, exactly
like the single-channel demo does for its one video.
"""
myVid_channels = [infoflow_scripts.gaussian_video_pyramid(myVid_rgb[..., c], scales=[8], sigma=1)[0]
                   for c in range(3)]
print(f"Downsampled channel shape: {myVid_channels[0].shape}  (T,H,W)")

# =============================================================================
#     2. run through the joint multichannel causal measures
# =============================================================================
# DDC/LK/correlation/GC each expose a "joint" pair of functions that compute
# every (source,target) channel combination from one shared pass over windows.
ddc_flows  = infoflow_scripts.causal_flow_multichannel_joint_ddc(myVid_channels, winsize=3, eps=1e-12, alpha=1e-2)
lk_flows   = infoflow_scripts.causal_flow_multichannel_joint_lk(myVid_channels, winsize=3, eps=1e-12)
corr_flows = infoflow_scripts.causal_flow_multichannel_joint_corr(myVid_channels, winsize=3, lag=1)
gc_flows   = infoflow_scripts.causal_flow_multichannel_joint_gc(myVid_channels, winsize=3, lag=1, alpha=.1)

# PCCA can't share one computation across all channel pairs (see the joint
# DDC/LK/corr/GC docstrings in InfoFlow/infoflow.py for why), so it instead
# reuses the older per-candidate-position confound-aware driver with the
# native (non-residualized) confound-aware PCCA measure.
pcca_flows = infoflow_scripts.compute_channel_flows(
    myVid_channels, pcca_cause_confound_native, pairs=[(0, 0), (0, 1)], winsize=3, scores=False,
    k=1, m=3, eta_xt=5e-4, eta_yt=5e-4, eta_xtkm=5e-4,
)

flow_dicts = {
    'DDC (joint)':  ddc_flows,
    'LK (joint)':   lk_flows,
    'Corr (joint)': corr_flows,
    'GC (joint)':   gc_flows,
    'PCCA (native confound-aware)': pcca_flows,
}

# =============================================================================
#     3. Visualization -- within-channel (0,0) and cross-channel (0,1) per method
# =============================================================================
xy_coords = np.indices(myVid_channels[0].shape[1:]); xy_coords = xy_coords.transpose(1, 2, 0)
xy_coords = xy_coords[..., ::-1]

sampling = 1
bg = myVid_channels[0][myVid_channels[0].shape[0] // 2]
bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

for method_name, flows in flow_dicts.items():
    for (src, tgt) in [(0, 0), (0, 1)]:
        flow = flows[(src, tgt)]
        label = f"{method_name}: {channel_names[src]} -> {channel_names[tgt]}"
        tag = f"{method_name.split()[0]}_{channel_names[src]}to{channel_names[tgt]}"

        plt.figure(figsize=(10, 10))
        plt.title(label)
        plt.imshow(bg, cmap='gray')
        plt.quiver(xy_coords[::sampling, ::sampling, 0],
                   xy_coords[::sampling, ::sampling, 1],
                   flow[::sampling, ::sampling, 1],   # x
                   -flow[::sampling, ::sampling, 0])  # y
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{tag}_quiver.png"), dpi=130)
        plt.close()

        flow_color = flow_to_color(flow[..., ::-1])
        plt.figure(figsize=(10, 10))
        plt.title(label + ' (flow-color)')
        plt.imshow(bg, cmap='gray')
        plt.imshow(flow_color, alpha=0.7)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{tag}_flowcolor.png"), dpi=130)
        plt.close()

        print(f"  {label}: mean(y,x) = "
              f"({np.nanmean(flow[..., 0]):.4f}, {np.nanmean(flow[..., 1]):.4f})  "
              f"-> saved {tag}_quiver.png / {tag}_flowcolor.png")

print(f"\nAll figures saved under {OUTDIR}/")
