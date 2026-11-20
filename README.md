# Restore, Then Cover: Restoration-Boosted Ensembles for Conformal Prediction under Image Corruption

Code Implementation for the paper _Restore, Then Cover: Restoration-Boosted Ensembles for Conformal Prediction under Image Corruption_

## What this code does

Conformal prediction gives an image classifier _prediction sets_ that contain the true label with a chosen
probability (e.g. 90%). This guarantee breaks when test images are degraded (noise, blur, JPEG, bad exposure).

**RBE-CP** (restoration-boosted ensemble conformal prediction) fixes this without any training:

1. the frozen classifier sees **13 views** of the test image (the image, its mirror, 6 restorers such as denoise or brighten, and 5 degrading probes);
2. the view probabilities are averaged into an ensemble `p̄`;
3. the best _restored_ view can raise a label by a fraction `λ` of its extra evidence:

   `p̃_y(x) = p̄_y(x) + λ · max(0, max_{restorers T} p_y(Tx) − p̄_y(x))`

4. `λ ∈ {0, ¼, ½, ¾, 1}` is chosen on synthetic corruptions of the calibration data (never on test data), and the
   threshold is calibrated on clean images, so coverage on clean data is guaranteed.

Every number in the paper is produced by the scripts below from logged results.

## Contents

```
configs/core/    frozen protocol, model lists, and one job list (JSON) per experiment
src/tracecal/    the Python package: datasets, models, views, feature extraction, RBE-CP and baselines
scripts/         job runner, evaluations, pre-registration reports, tables and figures
tests/           unit tests
audits/          pre-registrations (gates_v*.yaml) and the results reported in the paper (gates_v*_results.json)
```

---

## 1. Install

Requirements: Linux, **Python 3.10**, an NVIDIA GPU with a CUDA 12 driver (PyTorch 2.5.1, CUDA 12.4 wheels).
The tests and all evaluation steps after feature extraction run on CPU.

Run every command from the project folder.

### Option A: uv (recommended, installs the exact versions of `uv.lock`)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # only if uv is not installed
uv sync --extra dev --extra paper                   # creates .venv/ with all dependencies
uv run pytest -q -m "not gpu"                       # unit tests (about 1 minute)
```

### Option B: python venv + pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,paper]" --extra-index-url https://download.pytorch.org/whl/cu124
pytest -q -m "not gpu"
```

With option B, remove `uv run` from the commands below (for example `tracecal --help`, `python scripts/...`).
The job lists in `configs/core/` call `uv run` internally, so for the full pipeline (section 4) also run
`pip install uv`.

---

## 2. Set the paths

```bash
export PROJECT_ROOT=$(pwd)
export SHARED_DATASETS_ROOT=/path/to/datasets       # CIFAR and ImageNet datasets (layout below)
export DATA_ROOT=$PROJECT_ROOT/data                 # ImageNet-ES and the generated ImageNet-C-bar
source configs/env.sh
mkdir -p results/logs
```

Put these lines in a file and `source` it in every new shell.

### Datasets

Download the datasets from their official sources and place them like this:

```
$SHARED_DATASETS_ROOT/
  cifar-10-batches-py/           CIFAR-10 (python version)
  cifar-100-python/              CIFAR-100 (python version)
  CIFAR-10-C/   CIFAR-100-C/     corrupted CIFAR test sets
  Imagenet/                      ImageNet-1k from Hugging Face (arrow/parquet files; only the validation split is read)
  ImageNet-C/                    <corruption>/<severity>/...
  imagenet-r/   imagenet-a/      ImageNet-R and ImageNet-A
  clip_benchmark/imagenetv2/     ImageNet-V2       (clip_benchmark format)
  clip_benchmark/imagenet_sketch/ ImageNet-Sketch  (clip_benchmark format)

$DATA_ROOT/
  imagenet_es/ImageNet-ES/es-val/   ImageNet-ES (real camera captures)
  imagenet_es/ImageNet-ES/es-test/
  imagenet_c_bar/                   generated in step 4.4
```

Check what is found:

```bash
uv run tracecal data verify
```

### Model weights

- **ImageNet models** (timm, OpenCLIP): downloaded automatically on first use into `.cache/`.
- **CIFAR models**: trained by step 4.1.
- **Two robust ResNet-50s** (AugMix, DeepAugment) come from RobustBench (ImageNet, `corruptions` threat model). Put
  them at `.cache/robust/Hendrycks2020AugMix.pt` and `.cache/robust/Hendrycks2020Many.pt`; their SHA-256 is checked
  against `configs/core/models_v5.yaml`.

---

## 3. Quick check

```bash
uv run pytest -q -m "not gpu"      # unit tests
uv run tracecal --help             # all commands of the package
```

Minimal illustration of the method (NumPy only; the code used for the paper is
`src/tracecal/analysis/frontier_v3.py`):

```python
import numpy as np

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def rbe_probs(logits, restorers, lam):
    """logits: (n_images, 13 views, n_classes); restorers: indices of the restorer views."""
    pbar = softmax(logits.mean(axis=1))                     # test-time ensemble
    best = softmax(logits[:, restorers]).max(axis=1)        # best restored view, for every label
    return pbar + lam * np.clip(best - pbar, 0, None)       # RBE-CP probability; score = 1 - probability

def calibrate(p_cal, y_cal, alpha=0.1):                     # on CLEAN calibration images
    s = np.sort(1 - p_cal[np.arange(len(y_cal)), y_cal])
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return s[k - 1]                                         # threshold q_hat

def prediction_sets(p_test, q_hat):
    return (1 - p_test) <= q_hat                            # (n_images, n_classes): True = label in the set
```

---

## 4. Full pipeline

How it works:

- **GPU steps** use `scripts/gpu_queue.py`, which runs a job list from `configs/core/` with one job per GPU. List
  your GPUs with `--gpus` (for example `--gpus 0` for one GPU). Logs go to `results/logs/`. A finished job writes a
  `.ok` marker and is skipped if you run the queue again, so an interrupted queue can simply be restarted.
- **CPU steps** read the extracted features and write results to `results/` and `audits/`.

```bash
GPUS=0,1,2        # the GPUs to use
```

### 4.1 Train the CIFAR models (GPU)

ResNet-18 and ViT-S/16 on CIFAR-10 and CIFAR-100.

```bash
uv run python scripts/gpu_queue.py configs/core/jobs_train.json    --gpus $GPUS   # development models, seeds 0-4
uv run python scripts/gpu_queue.py configs/core/jobs_train_v3.json --gpus $GPUS   # fresh models, seeds 5-8
# a single model:  uv run tracecal train-cifar cifar10 resnet18 5
```

### 4.2 Extract the features (GPU)

One pass of each model over the 13 views of every image (clean, synthetic corruptions, test sets).

```bash
uv run python scripts/gpu_queue.py configs/core/jobs_extract_cifar.json    --gpus $GPUS
uv run python scripts/gpu_queue.py configs/core/jobs_extract_imagenet.json --gpus $GPUS
uv run python scripts/gpu_queue.py configs/core/jobs_extract_v3.json       --gpus $GPUS
uv run python scripts/gpu_queue.py configs/core/jobs_extract_v4.json       --gpus $GPUS
uv run python scripts/check_features.py --deep                               # integrity check
```

### 4.3 Main evaluation (CPU)

```bash
# v2: development study (restorative union, CIFAR seeds 1-4 and 5 ImageNet models)
uv run python scripts/run_cpu_jobs.py frontier-cifar --workers 4
uv run python scripts/imagenet_pipeline.py --workers 3
uv run python scripts/report_gates_v2.py all

# v3: RBE-CP confirmed on fresh models (main result of the paper)
uv run python scripts/run_cpu_jobs.py v3 --workers 4
uv run python scripts/report_gates_v3.py

# v4: symmetric-anchor variant (failed its main test; reported in the supplement)
for m in in_densenet121 in_mnv3_l in_deit3_s in_maxvit_t in_mixer_b16 in_deit_s in_effnet_b0 in_regnety16 in_convnext_s in_vit_s16; do
  uv run tracecal frontier-v4 $m imagenet && uv run tracecal frontier-v4-naturals $m
done
for t in c10_resnet18 c10_vits16; do for s in 5 6 7 8; do uv run tracecal frontier-v4 ${t}_s$s cifar10; done; done
for t in c100_resnet18 c100_vits16; do for s in 5 6 7 8; do uv run tracecal frontier-v4 ${t}_s$s cifar100; done; done
uv run python scripts/report_gates_v4.py
uv run python scripts/report_v3_on_v4_models.py          # frozen RBE-CP on the 5 v4 models (held-out)
```

### 4.4 Held-out evaluations of the frozen method

```bash
# v5: 7 large and robustly trained ImageNet models
uv run python scripts/gpu_queue.py configs/core/jobs_v5.json --gpus $GPUS
uv run python scripts/report_gates_v5.py

# v5b: ImageNet-C-bar. First clone the official generator (facebookresearch/augmentation-corruption)
# into .cache/src/augmentation-corruption, then generate the images:
uv run --with scikit-image python scripts/make_imagenet_c_bar_subset.py --workers 12
uv run python scripts/gpu_queue.py configs/core/jobs_v5b.json --gpus $GPUS
uv run python scripts/report_gates_v5b.py

# v5d: ImageNet-ES, real camera captures (the fixed restorers fail here; reported)
uv run python scripts/gpu_queue.py configs/core/jobs_v5d.json --gpus $GPUS
uv run python scripts/report_gates_v5d.py
```

### 4.5 Camera-aware RBE-CP (v6)

```bash
uv run python scripts/gpu_queue.py configs/core/jobs_v6dev.json --gpus $GPUS    # design, development models only
bash scripts/run_v6.sh                                                           # confirmation on 10 fresh models + report
uv run python scripts/gpu_queue.py configs/core/jobs_v6sec.json --gpus $GPUS    # 7 large / robust models (report only)
for m in in_rn50_tv1 in_augmix_rn50 in_deepaug_rn50 in_swin_b in_convnext_l in_vit_l16 in_clip_l14; do
  uv run tracecal frontier-v6 $m
done
uv run python scripts/report_gates_v6.py
uv run python scripts/gpu_queue.py configs/core/jobs_v6nat.json --gpus $GPUS    # natural shifts (not pre-registered)
uv run python scripts/report_v6_naturals.py
```

### 4.6 Ablations and additional analyses (supplement)

```bash
bash scripts/run_all_ablations.sh                       # component ablations
bash scripts/run_all_majority.sh                        # majority vote over the per-view sets
bash scripts/run_e2.sh                                  # sensitivity to the restorer magnitudes

# Proposition 3 bound vs measured coverage
for m in c10_resnet18_s5 c10_resnet18_s6 c10_resnet18_s7 c10_resnet18_s8 c10_vits16_s5 c10_vits16_s6 c10_vits16_s7 c10_vits16_s8 \
         c100_resnet18_s5 c100_resnet18_s6 c100_resnet18_s7 c100_resnet18_s8 c100_vits16_s5 c100_vits16_s6 c100_vits16_s7 c100_vits16_s8 \
         in_effnet_b0 in_regnety16 in_vit_s16 in_densenet121; do
  uv run tracecal extract-e1 $m
done
uv run python scripts/report_e1_bound.py

# gains by severity and per corruption
for m in in_deit_s in_effnet_b0 in_regnety16 in_convnext_s in_vit_s16 in_densenet121 in_mnv3_l in_deit3_s in_maxvit_t in_mixer_b16; do
  uv run python scripts/report_severity_tail.py $m
done
for m in in_deit_s in_effnet_b0 in_regnety16 in_convnext_s in_vit_s16; do
  uv run python scripts/report_v3_per_corruption.py $m
done
uv run python scripts/report_v3_robustness.py

# TTA-CP with its own augmentation policies, other baselines, cost
uv run python scripts/gpu_queue.py configs/core/jobs_ttacp.json --gpus $GPUS
uv run python scripts/report_ttacp_policies.py
uv run python scripts/report_ecp.py
CUDA_VISIBLE_DEVICES=0 uv run python scripts/measure_latency.py
uv run python scripts/explore_es_views.py               # why test-time views fail on ImageNet-ES (exploratory)

# secondary checks on CIFAR (grid density, empty sets, mixed streams)
for a in dense nonempty mixed; do uv run python scripts/run_cpu_jobs.py $a-cifar --workers 2; done
uv run python scripts/report_dense_check.py cifar
uv run python scripts/report_nonempty.py cifar
uv run python scripts/report_mixed_stream.py cifar
```

### 4.7 Tables and figures

Run in this order (the first command creates `paper/tables/`).

```bash
uv run python scripts/make_v2_tables.py
uv run python scripts/report_per_corruption.py
uv run python scripts/make_v3_tables.py
uv run python scripts/make_v4_tables.py
uv run python scripts/make_v5_tables.py
uv run python scripts/make_ablation_tables.py
for w in frontier teaser restorers bound; do uv run python scripts/make_v2_figures.py $w; done
uv run python scripts/make_teaser.py inc in_effnet_b0 fog:4 2027 v2    # Figure 1
uv run python scripts/make_fig_realcam.py                              # real camera figure
uv run python scripts/make_fig_heldout.py                              # held-out models
uv run python scripts/make_fig_forest.py                               # every pre-registered comparison
uv run python scripts/make_fig_uncurated.py inc
uv run python scripts/make_fig_uncurated.py es
uv run python scripts/make_qualitative.py
```

---

## 5. Where the results are

| Output                                      | Location                       |
| ------------------------------------------- | ------------------------------ |
| Extracted features (logits of the 13 views) | `results/features/`            |
| Per-model evaluations                       | `results/raw/`                 |
| Summary results (JSON)                      | `results/tables/`              |
| Pre-registered test results                 | `audits/gates_v*_results.json` |
| LaTeX tables                                | `paper/tables/`                |
| Figures (PDF, PNG)                          | `results/figures/`             |
| Logs                                        | `results/logs/`                |

The files in `audits/` are the results reported in the paper. Re-running a report overwrites them: compare with
`git diff audits/`.

### The studies (each one was pre-registered in `audits/` before its data was analysed)

| Study  | File             | What it tests                                      | Outcome                   |
| ------ | ---------------- | -------------------------------------------------- | ------------------------- |
| v2     | `gates_v2.yaml`  | restorative union (development)                    | partly failed (ImageNet)  |
| **v3** | `gates_v3.yaml`  | **RBE-CP on fresh CIFAR and ImageNet models**      | **passed**                |
| v4     | `gates_v4.yaml`  | symmetric-anchor variant                           | failed main test; v3 kept |
| v5     | `gates_v5.yaml`  | frozen RBE-CP on 7 large / robust ImageNet models  | passed                    |
| v5b    | `gates_v5b.yaml` | ImageNet-C-bar (corruptions unlike ImageNet-C)     | passed                    |
| v5d    | `gates_v5d.yaml` | ImageNet-ES real camera, fixed restorers           | failed (reported)         |
| **v6** | `gates_v6.yaml`  | **camera-aware restorers on real camera captures** | **passed**                |

Ablations were declared in `ablations_v5c.yaml` and the TTA-CP baseline check in `baseline_ttacp_policies.yaml`,
both before any number.

---

## Licences

Code: MIT (see `LICENSE`). Model weights and datasets are not redistributed; they keep their own terms:

| Asset                                              | Terms                                         |
| -------------------------------------------------- | --------------------------------------------- |
| CIFAR-10/100                                       | MIT-style terms                               |
| CIFAR-10/100-C, ImageNet-C, ImageNet-ES            | CC BY 4.0                                     |
| ImageNet-C-bar generation code                     | MIT                                           |
| ImageNet                                           | research terms                                |
| ImageNet-V2, ImageNet-Sketch, ImageNet-R/-A        | MIT                                           |
| timm weights                                       | per model card                                |
| OpenAI CLIP                                        | MIT                                           |
| RobustBench AugMix / DeepAugment ResNet-50 weights | per RobustBench and the original repositories |
