# Line-laser stripe centre extraction: per-row regression network (v3)

Scene-general sub-pixel centre extraction for line-laser stripes. For every image row the network predicts a softmax over x (W pixels plus one "no stripe in this row" class); the centre is then obtained with soft-argmax and a local centroid. This repository contains the code, representative training data, training logs, evaluation results, and (as release assets) the model weights.

> Status: research prototype. There is no trustworthy real-world ground truth, so every claim below is labelled with its basis and limitations.

## 1. Method overview

- Model: U-Net from `segmentation_models_pytorch` (resnet18, trained from scratch), 4 output channels (row head + column head), 12.46M parameters. About 8.6 ms/frame at 1080p on GPU (bf16, measured on v2).
- Decoding: top-3 peaks per row, +/-3 px centroid, Viterbi smoothing. `OrientRegressor` selects the stripe orientation automatically from the peak score.
- Loss: cross-entropy against a Gaussian (sigma = 1) centred on the label, plus 0.5 x L1 on the soft-argmax inside a +/-3 px window.
- Training data: roughly 50% synthetic, 50% real wood frames.
  - Synthetic: generated on the fly with random laser colour, width (0.8-5.5 px), amplitude (30-900), procedural or real (inpainted wood) backgrounds, up to 4 distractor stripes, glare blobs, occluders and 8% empty images. Labels are exact (x position / -1 for absent / NaN for ignore). CPU version: `synth3.py`; GPU version: `synth_gpu.py` + `core3g.py`.
  - Real: `data/real_train/data_scan_v3` (598 train + 120 val frames, wood only). Labels are the rows where Steger and the mask agree, so label accuracy is bounded by those methods (circular).

## 2. Layout

```
code/colreg/      training, inference, synthesis and evaluation scripts (train3.py, train3g.py, infer.py, synth*.py, bench_*.py, ...)
code/misc/        helper scripts from the earlier segmentation stage
code/board_eval/  evaluation on calibration-board images (line / plane residuals)
data/real_train/  data_scan_v3 (real frames + masks), bg (150 inpainted backgrounds), labels (per-row label npz), hand_labels
data/eval/        cam_snaps (16 plastic helmet / cup frames), board25 (25 images of the laser on a ChArUco board)
logs/             per-run log.json / meta.json and training logs
results/          benchmark and comparison figures
docs/             earlier progress report (report.md, in Chinese; some results are superseded)
release_assets/   weights (11 .pt files, ~48 MB each) and SHA256SUMS.txt -- not tracked in git, see section 6
setup.sh          links the data to where the scripts expect it and places the weights
```

## 3. Reproducing

Requirements: Python, PyTorch (CUDA), torchvision, `segmentation-models-pytorch`, OpenCV, NumPy.

```bash
./setup.sh                      # links data, copies release_assets/*.pt to code/colreg/runs/<name>/last.pt
cd code/colreg
python train3.py  --name r18_v3  --iters 16000               # CPU synthesis (~27 min on one GPU)
python train3g.py --name r18_v3g --iters 16000 --gpu_synth   # GPU synthesis
python bench_eval.py --nosteger --models name=runs/r18_v3/last.pt:orient
```

Main options are in `train3g.py`: `--bs 8 --lr 1e-3 --p_synth 0.5 --p_rot 0.5 --seed 0`. Ablation settings are in `run_abl.sh` (8000 iterations each). Some scripts still contain the author's `~/stripe_seg/...` paths and need adjusting to your layout.

Inference:

```python
from infer import OrientRegressor
m = OrientRegressor("runs/r18_v3/last.pt")
pts = m.extract(bgr_image)[:, :2]    # N x 2, sub-pixel (x, y)
```

## 4. Results and how much to trust them

### Synthetic benchmark (12 difficulty axes x 5 levels, fixed seeds)
- v3 fixes colour (red-laser miss rate 23% -> 0), low contrast (amplitude 30: 100% missed -> 0) and orientation (v2 missed ~85% when rotated -> 0).
- Remaining problems: distractor stripes (~8% false points; convention is "strongest stripe = target") and occlusion (~7% false points).
- Caveat: the benchmark is drawn from the same family of generators as the training data, so the gains are overstated.

### Ablations (8000 iterations; benchmark 20 images per level + 7 hand-labelled frames)
- Pretrained encoder, full-resolution skip, synthetic-only: no gain beyond seed noise.
- Segmentation head is worse (50 false points per empty image on average); real-only data is clearly the worst; 16000 iterations halves distractor false positives compared with 8000.
- GPU-synthesised data (v3g) matches the CPU version (v3) on validation, and is slightly worse on benchmark distractors (12.6% vs 7.9%); one seed each, so equivalence is not established.
- The hand labels cover only 7 frames with 2-3 px click noise, and the coverage metric does not penalise false points.

### Laser on a ChArUco board (25 frames, `code/board_eval`)
The black/white squares make the stripe discontinuous. NN centres were back-projected onto the PnP board plane and compared with Steger:

| | Steger | v3 | v3g |
|---|---|---|---|
| Points lying on the fitted line | 34% | 99% | 99% |
| Valid points per frame | 131 | 476 | 475 |
| Line residual RMS (mm) | 0.91 | 0.31 | 0.33 |
| Plane-fit RMS (mm) | 1.01 | 0.43 | 0.44 |

- Fitting the laser plane on 20 frames and testing on the other 5 gives a held-out RMS of 0.45 mm for the NN, close to the full-data fit.
- The NN plane and the Steger plane differ by 0.14 deg in normal and about 0.5 mm in offset at the stripe centroid; there is no ground truth to say which is closer to the true plane.
- Steger was run with the calibration package's blue-tuned parameters, not retuned for these images, so part of the 3x gap comes from an untuned baseline.
- Line residual favours smooth outputs and cannot reveal a global offset (e.g. locking on the halo rather than the core).
- 25 frames, one camera, one seed per model; the training data contains no checkerboards.

### Wooden sphere scan (V03_150, 37 frames, sphere-fit residual)
Steger 0.43 mm, NN v3 0.37 mm (fitted radius 49.65 / 49.76 mm). The difference is near the noise level and was not tested over multiple seeds. A sphere fit absorbs global translation and scale errors, so it does not measure absolute accuracy.

## 5. Known limitations

- No real-world ground truth; training labels and several evaluations share the same rules, so they are partly circular.
- Real training frames are wood only; plastic parts (`cam_snaps`) are used for evaluation only.
- Metal and strongly specular scenes are untested.
- The real benefit for laser-plane calibration still needs to be confirmed by measuring an object of known size with the new plane; this has not been done.
- Hand-eye calibration error (about 2 mm on validation) and timestamp synchronisation are outside the scope of this work and may mask improvements in stripe extraction.

## 6. Weights

The 11 `.pt` files (v3 / v3g are the main models; the rest are ablations and earlier versions) are attached to a GitHub Release; checksums are in `release_assets/SHA256SUMS.txt`. Download them into `release_assets/` and run `./setup.sh`.

| File | Description |
|---|---|
| r18_v3.pt | main model, CPU synthesis, 16k iterations |
| r18_v3g.pt | GPU synthesis, 16k iterations |
| r18_v2.pt | v2, 12k iterations, expects near-vertical stripes |
| r18_scratch_s0.pt | v1, 6k iterations |
| abl_*.pt | ablations (8k iterations): base_s0/s1, pre, seg, fullres, synth, real |

## 7. Data notice

`data/` contains wood scan frames, plastic-part photos and calibration-board images captured by the author. Make sure you are entitled to use or redistribute them before doing so.
