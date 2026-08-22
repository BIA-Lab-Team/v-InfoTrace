# v-InfoTrace (Multiscale Pixel Spatiotemporal Information Flows)
<p align="center">
  <img src="img/concept_figure.jpg" width="800"/>
  <!-- <img src="https://github.com/fyz11/MOSES/blob/master/Example_Results/mesh_frame20_red.png" width="300"/> -->
</p>
v-InfoTrace (multiscale pixel spatiotemporal information flows) is a formal modelling framework to systematically apply 1D causal measures to image pixels. Neighborhood pixels are naturally correlated spatiotemporally which complicates direct application of 1D causal measures. Here we model the NxN(xN) pixel neighborhood as a rigid body and demonstrate how to condition to extract the dense pixel-to-pixel information transfer in videos using any desired 1D causal measure, and in a multiscale manner. The package supports both **2D+time** and **3D+time** (volumetric) data, and both single-channel and **multichannel** video, computing within-channel and cross-channel pixel-to-pixel information flow. In our paper we show how we can discover salient pixel-to-pixel information highways in videos of diverse phenomena spanning traffic and crowd flow, collision physics, fish swarming, moving camouflaged animals, human action, embryo development, cell division and cell migration.<br>

## Contents
- [News](#news)
- [Supported 1D causal measures](#supported-1d-causal-measures)
- [Different information flow extracted from diverse videos compared to optical flow](#different-information-flow-extracted-from-diverse-videos-compared-to-optical-flow)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Example scripts](#example-scripts)
- [Issues](#issues)
- [BIA Lab Links](#bia-lab-links)

## News
- **v-InfoTrace** derives from the original **u-InfoTrace** and extends it with (1) **multichannel** support — computing both within-channel and cross-channel (confound-aware) causal flow across an arbitrary number of channels, for all five causal measures (DDC, LK, correlation, PCCA, GC), and (2) a **3D** generalization of the full pipeline, extending every causal measure and driver from 2D+time videos to 3D+time (Z,Y,X+time) volumes.

For more information on the original methodology this package builds on, please read our paper, [**Multiscale Pixel Spatiotemporal Information Flows**](https://openreview.net/forum?id=4P0qQrU_SlN), *A causal view on dynamical systems, NeurIPS 2022 workshop*, written by Felix Yuran Zhou, Roshan Ravishankar.

If you use the code or think our work is useful in your research, please consider citing:

```
@INPROCEEDINGS{zhou2022multiscale,
	AUTHOR = {Zhou, Felix Yuran and Ravishankar, Roshan},
	TITLE = {Multiscale Pixel Spatiotemporal Information Flows},
	BOOKTITLE = {A causal view on dynamical systems, NeurIPS 2022 workshop},
	YEAR = {2022},
}
```

## Supported 1D causal measures
Any 1D causal measure can in principle be plugged into the framework. v-InfoTrace currently provides five:

- **Conditional Granger Causality (cGC)** — tests whether a candidate pixel-neighborhood's past (and, in the joint multichannel driver, present) values improve prediction of a target's future beyond what the target's own past already explains, via nested reduced-vs-full autoregressions and a log-variance-reduction score.
- **Differential Covariance (DDC)** — fits a single linear model relating a stacked "current state" to its own time-derivative, and reads directed pixel-to-pixel influence off the resulting coefficient (cross-covariance) matrix.
- **Linear LK Flow** — a linear Kalman-filter-style state-space model over the same stacked pixel-neighborhood state, with directionality read from the fitted transition/covariance structure, analogous to DDC but under a different (state-space) linear model.
- **Partial Canonical Correlation Analysis (PCCA)** — finds the linear projections of a candidate and target pixel-neighborhood's histories that are maximally correlated, after partialling out the target's own lagged history, via a generalized eigenvalue problem over partial covariances.
- **Partial Directed Coherence / Directed Transfer Function (PDC/DTF)** — fits a multivariate autoregressive (MVAR) model over the pixel-neighborhood and derives frequency-domain directed-connectivity measures (PDC, DTF) from the resulting AR coefficients.

**Note:** multichannel support for **PCCA** and **PDC/DTF** is still work-in-progress — cGC, DDC, and Linear LK Flow have complete, validated multichannel drivers (`InfoFlow/infoflow.py`); PCCA and PDC/DTF currently only have confound-aware single-pair building blocks (`pcca_cause_confound_native`, `PDC_central_flow_confound`) that are not yet wired into a validated multichannel flow driver.

## Different information flow extracted from diverse videos compared to optical flow
<p align="center">
  <img src="img/multi_example_flow.jpg" width="800"/>
</p>
Compared to optical flow, information flows (right columns from cGC onward) capture salient patterns and reveal 'information highways'. Depending on the modelling assumptions of the 1D causal measure, we find information flow highlights different attributes of the pixel-to-pixel relationship in videos. 

### Dependencies
v-InfoTrace relies on the following excellent packages (installed automatically via `pip install .`, or see `requirements.txt`/`pyproject.toml` for exact version pins):
- [numpy](https://numpy.org/) (`<2`)
- [scipy](https://www.scipy.org/)
- [scikit-image](https://scikit-image.org/)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [matplotlib](https://matplotlib.org/)
- [opencv](https://pypi.org/project/opencv-contrib-python/) (`==4.5.*`)
- [tqdm](https://tqdm.github.io/)
- [joblib](https://joblib.readthedocs.io/)
- [contourpy](https://contourpy.readthedocs.io/) (`<=1.2.0`)
- [imagecodecs](https://pypi.org/project/imagecodecs/) (`<2025`)

### Installation
The package can be installed after cloning this repository using the following command.
```
pip install .
```
The package is also available on PyPI and can be installed directly via:
```
pip install v-InfoTrace
```
The package is verified to work with **Python 3.9–3.12** on **Windows, macOS (Intel and Apple Silicon), and Linux (x86_64 and aarch64)**. Python 3.13+ is not currently supported (the `numpy<2` pin has no 3.13 wheel).

### Example scripts
- `2023-03-31_testScript_InfoFlow_shorter.py` demonstrates how to extract the single-channel 2D information flow using a variety of 1D causal measures, for a video from the crowdflow dataset (https://www.crcv.ucf.edu/research/data-sets/crowd-segmentation/) downsampled by a factor of 8.
- `2026-08-22_exampleScript_InfoFlow_multichannel.py` demonstrates the **multichannel** extension — computing within-channel and cross-channel information flow across the raw RGB channels of a crowdflow video.
- `2026-08-22_exampleScript_InfoFlow_3D.py` demonstrates the **3D** extension — recovering a known drift direction from a synthetic 3D+time volume using all five causal measures.

### Issues
If issues arise, please contact Felix Zhou (felix.y.zhou@vanderbilt.edu) or raise a GitHub issue on this repository.

### BIA Lab Links
[BIA Lab Website](https://bia-lab-team.github.io/BIA_Lab_Website/)

[Software Links](https://github.com/BIA-Lab-Team/)
