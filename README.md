# ScienceQA: early and late fusion for answer prediction

COMP1013 Multi-Modal Data Analysis, Project 2.2. Qingyan Han, a1988836.
This is the individual ScienceQA project developed from Project 2.1.

The project compares frozen visual/text representations with early feature fusion and late probability fusion. It uses the official ScienceQA splits, restricted to questions with an image and 2–4 answer options. Ground-truth lectures and solutions are never model inputs. See [PROTOCOL.md](PROTOCOL.md) for the specification fixed before execution and [AI_ASSISTANCE.md](AI_ASSISTANCE.md) for the actual assistance record.

Coursework repository: https://github.com/QingyanHan/comp1013-scienceqa-2026.

## Measured results

One locked test evaluation, 1,979 questions, primary seed 42:

| Method | Accuracy | Macro-F1 | NLL | Brier |
|---|---:|---:|---:|---:|
| Majority A | 35.67% | 0.131 | 23.185 | 1.287 |
| TF-IDF + LR | 36.53% | 0.226 | 1.097 | 0.655 |
| Image only | 37.04% | 0.325 | 1.097 | 0.659 |
| Text only | 43.86% | 0.434 | 1.071 | 0.638 |
| Early fusion | 42.40% | 0.407 | 1.073 | 0.644 |
| Late fusion | 43.36% | 0.428 | 1.055 | 0.629 |

Fusion does not improve primary accuracy over text. Late fusion improves probability scores. Three-seed mean accuracies are text 43.93%, early 42.95%, late 43.20%. These are optimisation repeats on the same splits. Exact images/texts repeat across splits; read the overlap diagnostics before making generalisation claims. See `results/independent_verification.json` for the recorded 136-check audit.

## Contents

- `run_project.py`: staged CPU pipeline, dataset audits, encoders, model selection, evaluation.
- `config.json`: fixed hyperparameters and source revisions.
- `results/`: row manifests, selection and evaluation receipts, predictions, metrics, timings and diagnostics.
- `verify_results.py`: independent metric and checksum verification without training.
- `analyse_results.py`: descriptive analysis and figures from saved results.
- `prepare_inputs.py`: optional downloads for a new machine. Never invoked by the experiment runner.
- `reproduce.ps1`: rerun into a new directory without overwriting the evidence.
- `delivery/`: report, slides, speaking script and presentation materials.
- `models/` and `cache/`: local generated files, excluded from Git. Raw data and pretrained weights are not distributed.

## Recorded run and environment

Recorded on Windows 11, Python 3.14.0, using CPU and four threads. CUDA-capable packages were installed but all models and tensors use CPU. Exact package versions are in `results/environment.json` and `requirements.txt`. Those files describe the measured environment; a CPU wheel installation is an alternative, not an independently verified environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
```

For the exact installed CUDA-tagged wheels, use `requirements.txt` with the PyTorch cu128 package index. No GPU is needed. Reproduction on other operating systems, BLAS versions or CPU wheels can differ numerically; compare the saved predictions and seed ranges rather than asserting bitwise equality.

## Inputs

The original run used only existing local caches. By default the code expects these paths below the current user's home:

```text
.cache/huggingface/datasets/derek-thomas___science_qa/default/0.0.0/
  f18b0a70359ebfb41f658fd564208d0355b013f4/science_qa-{train,validation,test}.arrow
.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/
  e8f8c211226b894fcb81acc59f3b34ba3efd5f42/
.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

For a new machine, this optional command downloads the pinned dataset and encoders, approximately 1.2 GB. Allow extra disk space for Arrow data and feature caches. It keeps existing files. An offline fixture test verifies Parquet-to-Arrow row order, embedded image bytes, missing images and pinned revisions. Live downloads and a clean-machine full reproduction have not been tested.

```powershell
python prepare_inputs.py --download
```

Alternatively point the following environment variables at existing inputs:

```powershell
$env:SCIENCEQA_DATA_DIR = 'D:\data\scienceqa-arrow'
$env:SCIENCEQA_TEXT_MODEL_DIR = 'D:\models\minilm-snapshot'
$env:SCIENCEQA_RESNET_WEIGHTS = 'D:\models\resnet18-f37072fd.pth'
```

The dataset adapter converts fixed upstream Parquet files to Arrow IPC streams in original order. Serialization hashes may differ from the original Hugging Face cache; row counts, ordered manifest IDs, answer labels and image/text hashes should match. The recorded source file hashes remain evidence for the original run.

## Reproduce without overwriting results

```powershell
.\reproduce.ps1 -OutputDirectory ..\scienceqa_reproduction_01
```

This copies the frozen implementation/configuration/protocol, then runs `prepare`, `features`, `train`, `evaluate`, verification, analysis and the selected-example token audit. Relative paths resolve against the current PowerShell location. Use a new directory that does not exist. Each training or evaluation stage is guarded against rerunning a completed locked stage. Failures remain in `logs/`; do not delete failed evidence to hide a repair.

Manual sequence in a new run directory:

```powershell
python -u run_project.py prepare
python -u run_project.py features
python -u run_project.py train
python -u run_project.py evaluate
python verify_results.py
python analyse_results.py
python audit_selected_examples.py
```

The first three stages use training/validation data only. `train` writes the selected model hashes before `evaluate` reads test data. Selection uses validation negative log-likelihood. Seed 42 is the predeclared primary analysis; seeds 43 and 44 measure optimisation sensitivity, not independent samples. Do not choose the best test seed or tune parameters after evaluation.

## Audit the existing result

Run `python -m unittest discover -s tests -v` for three offline helper regression tests: dataset conversion with mocked downloads, relative-path/overwrite handling with training mocked, and rejection of altered identifiers or NaN probabilities even under `python -O`. These engineering tests are separate from the recorded scientific experiment; they do not retrain or evaluate models.

The complete local delivery includes fitted heads/scalers for checksum verification. A source-only Git checkout excludes these binaries: reproduce into a new directory to regenerate them, or use the supplied local fitted-model package. `verify_results.py` checks source/model/prediction hashes, label alignment, legal-choice probabilities and independently recomputes all four reported metrics. `analyse_results.py` produces descriptive plots without fitting any model.

For a source-only checkout without local model binaries, `python verify_results.py --predictions-only` recomputes metrics and checks source/prediction hashes while explicitly skipping model-file checks. Its separate receipt does not replace the full recorded verification. The analysis figure generator additionally needs the original dataset to reproduce example images.

Late fusion uses a validation-selected scalar text weight. Early fusion combines standardised feature vectors. The image-only baseline is given the number of choices so it can mask impossible positions, but receives no question or choice text. Frozen image features do not provide OCR. Accuracy gains therefore do not establish general visual reasoning.

Exact image/text overlap across official splits is reported with unchanged denominators. Unseen-image/text subset results and shuffled-image diagnostics are descriptive. The bootstrap resamples questions and can underestimate uncertainty where templates repeat. Macro-F1 refers to answer positions A–D, not demographic fairness.

## Source and licensing attribution

- Dataset and original paper: [ScienceQA](https://github.com/lupantech/ScienceQA), Lu et al., NeurIPS 2022, [paper](https://arxiv.org/abs/2209.09513).
- Pinned conversion: [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), dataset card labels CC BY-SA 4.0. Images shown in the report are credited examples, not newly generated assets. Consult the upstream data licence for reuse.
- Image encoder: [torchvision ResNet-18 IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [He et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
- Text encoder: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [Sentence-BERT](https://aclanthology.org/D19-1410/), Reimers and Gurevych, 2019.

## Presentation and submission

The report and slides use the recorded results. Any synthetic narration is explicitly labelled as a rehearsal, not the student's own speech. The student must understand the work, review disclosures, record the required personal presentation if applicable, and attend the live Q&A. This repository does not perform a Canvas submission or confirm course-specific AI permissions. No inbox messages, account records or unrelated coursework belong in the repository.
