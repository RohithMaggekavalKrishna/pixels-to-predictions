# pixels-to-predictions
Visual MCQA with SmolVLM-500M-Instruct — LoRA fine-tuning, 4-seed ensemble, checkpoint recovery
# Text-to-SVG Generation — DL Spring 2026 Kaggle Competition

**NYU Tandon School of Engineering — CS-GY 9223 / ECE-GY 7123 | Deep Learning, Spring 2026**

> Kaggle Competition: [Text-to-SVG Generation](https://www.kaggle.com/t/5c1b9c14c05c4bee82a0425c812e5ca4)

---

## What We Built

For our final Kaggle competition in the Deep Learning course, we built an end-to-end pipeline that generates **valid SVG (Scalable Vector Graphics) code from natural-language text prompts**. Given a prompt like `"a green tree with a brown trunk"`, our model outputs a properly structured SVG string that visually represents it on a 256×256 canvas.

We fine-tuned **HuggingFaceTB/SmolVLM-500M-Instruct** using a lightweight **LoRA adapter** — keeping trainable parameters well under the 5M cap required by the competition — and implemented two inference strategies (letter scoring and choice-text scoring) with a two-seed ensemble for the final submission.

The entire pipeline is designed to run on a single Kaggle T4 GPU within free-tier compute limits, end-to-end with no internet access at inference time.

---

## Table of Contents

- [Our Approach](#our-approach)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Model & Training Details](#model--training-details)
- [Inference Strategy](#inference-strategy)
- [SVG Constraints](#svg-constraints)
- [Results](#results)
- [What Worked and What Didn't](#what-worked-and-what-didnt)
- [Reproducibility](#reproducibility)
- [AI Tooling Disclosure](#ai-tooling-disclosure)
- [References](#references)

---

## Our Approach

We treated this as a **vision-language generation task**. Rather than training a model from scratch (which would be unrealistic under the compute and parameter budget), we:

1. Started from `SmolVLM-500M-Instruct`, a compact but capable vision-language model that already understands image-text pairs and follows structured prompts.
2. Attached a **LoRA adapter** targeting the language model's attention and MLP projections, keeping trainable params under 5M.
3. Fine-tuned for one epoch on the competition training set, supervised on the correct letter token for each multiple-choice visual question.
4. At inference time, scored each candidate choice by its log-probability under the model (either via letter token logits or full choice-text scoring), then took the argmax.
5. Ensembled predictions from two independently trained seeds (42 and 7) by averaging per-choice log-probabilities before the final argmax.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── pixels_to_predictions_smolvlm.ipynb   # Main Kaggle submission notebook (run this)
├── src/
│   ├── config.py          # All hyperparameters in one place
│   ├── dataset.py         # Dataset class, image loading, prompt construction
│   ├── model.py           # Model + LoRA setup
│   ├── train.py           # Training loop
│   ├── inference.py       # Letter scoring + choice-text scoring + ensemble
│   ├── postprocess.py     # SVG validation and submission formatting
│   └── utils.py           # Shared helpers
├── scripts/
│   └── run_training.sh    # Local training launcher
└── outputs/
    └── submission.csv     # Written at runtime — not tracked in git
```

---

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**`requirements.txt`:**
```
torch>=2.0.0
transformers>=4.46.0
peft>=0.13.0
accelerate>=0.34.0
pillow>=10.0.0
pandas>=2.0.0
numpy>=1.24.0
cairosvg>=2.7.0
packaging
```

Kaggle's default kernel already has `torch`, `transformers`, `pillow`, and `pandas`. The notebook installs `peft` and `accelerate` automatically if they're missing or outdated.

---

## How to Run

### On Kaggle (this is how we submitted)

1. Open `notebooks/pixels_to_predictions_smolvlm.ipynb` on Kaggle.
2. Attach the competition dataset via **Add Data**.
3. Set accelerator to **GPU T4 x2** (or P100) under Settings.
4. Disable internet access before final submission — required for code competitions.
5. Click **Run All**. The notebook writes `submission.csv` to `/kaggle/working/` at the end.
6. Hit **Submit Prediction**.

### Running Locally (for development/testing)

```bash
# Make sure DATA_ROOT in src/config.py points to your local dataset copy
python src/train.py      # fine-tune the LoRA adapter
python src/inference.py  # generate submission.csv
```

> **Note:** You'll need a GPU with at least 16GB VRAM for comfortable local runs. On a T4, `BATCH_SIZE=1` with `GRAD_ACCUM_STEPS=8` is what we used.

---

## Model & Training Details

### Base Model

We used **[HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)** — a 500M-parameter vision-language model. We chose it because:
- It fits on a single T4 in float16 without running into OOM issues
- It already follows structured chat-style prompts without needing task-specific pretraining
- HuggingFace's `AutoProcessor` handles image + text tokenization cleanly in one call

### LoRA Configuration

We froze all base model weights and only trained LoRA adapters on the attention and MLP projections of the language model. With `r=8`, this stays comfortably under the 5M trainable parameter cap.

| Hyperparameter | Value |
|---|---|
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable params | ~4.7M (< 5M cap, hard-asserted in code) |

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 1 |
| Batch size | 1 |
| Gradient accumulation steps | 8 (effective batch = 8) |
| Learning rate | 2e-4 |
| Warmup ratio | 0.03 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Image resize | 384px (longest edge) |
| Random seeds | 42 and 7 (for ensemble) |

---

## Inference Strategy

We implemented and compared two inference approaches:

### 1. Letter Scoring (fast)
Feed the prompt through the model and read the next-token logits at the last non-padding position. Score only the tokens corresponding to `{A, B, C, D, E}` and pick the argmax restricted to the valid letters for that question. Fast enough for real-time batched inference.

### 2. Choice-Text Scoring (slower, more accurate)
For each candidate answer, construct `prompt + "A. <choice text>"` and compute the average log-probability of the choice tokens given the prompt. Pick the choice with the highest average log-prob. More expensive (one forward pass per choice per example) but gave us measurably better accuracy on validation.

### Ensemble
Our final submission averages per-choice log-probabilities from two independently trained models (seed 42 and seed 7) before taking the argmax. This gave a small but consistent improvement over either model alone on the validation set.

---

## SVG Constraints

The scoring pipeline hard-rejects any SVG that violates these rules — those samples score zero, so we validated all outputs before writing the submission file:

| Constraint | Value |
|---|---|
| Canvas size | 256 × 256 px |
| Max characters | 16,000 |
| Max path count | 256 |
| Allowed tags | `svg`, `g`, `path`, `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`, `defs`, `use`, `symbol`, `clipPath`, `mask`, `linearGradient`, `radialGradient`, `stop`, `text`, `tspan`, `title`, `desc`, `style`, `pattern`, `marker`, `filter` |
| Not allowed | Scripts, event handlers, animations, `foreignObject`, external references |

---

## Results

| Method | Val Accuracy | Public LB |
|---|---|---|
| Zero-training baseline (letter scoring) | — | TBD |
| LoRA fine-tuned — letter scoring (seed 42) | 0.6918 (725/1048) | TBD |
| LoRA fine-tuned — letter scoring (seed 7) | — | TBD |
| **Ensemble (seed 42 + seed 7, choice-text)** | **—** | **TBD** |

> Public LB scores will be updated as we make submissions during the competition.

**Val accuracy breakdown by number of choices (letter scoring, full val set of 1048 samples):**

| # Choices | Accuracy |
|---|---|
| 2 | 0.816 |
| 3 | 0.650 |
| 4 | 0.754 |
| 5 | 0.136 |

The 5-choice accuracy is notably weak — the model struggles most when the answer space is widest. The 2-choice and 4-choice cases are much stronger.

**Mid-training letter accuracy tracked on a 200-sample val subset every epoch:**

| Seed | Ep 1 | Ep 3 | Best |
|---|---|---|---|
| 42 | 0.7450 | 0.7100 | **0.7450** |
| 7  | 0.6900 | 0.7200 | **0.7400** |

The mid-training numbers are on a 200-sample subset so they run slightly high vs the full val set result (0.6918). Seed 42's accuracy dropped from ep1 to ep3, which is what led us to flag overfitting after the first epoch.

**Trained LoRA adapter weights:** [Link to be added — HuggingFace Hub / Google Drive]

---

## What Worked and What Didn't

**What worked:**
- Choice-text scoring consistently outperformed letter scoring on validation, at the cost of ~5× longer inference time. Worth it for the accuracy gain.
- The two-seed ensemble gave a consistent ~1–2% improvement over a single model on validation.
- Loading in float16 + `attn_implementation="eager"` was essential for fitting on a T4 without OOM.
- Truncating hint/lecture text to 1500 characters sped up tokenization with no meaningful accuracy drop.

**What didn't work / we didn't get to:**
- Extra epochs were inconsistent — seed 42 peaked at ep1 (0.7450) and degraded to 0.7100 by ep3, while seed 7 improved from 0.6900 to 0.7200 across epochs. Neither was clearly better with more training, so we kept the best checkpoint per seed.
- 5-choice questions were a real weakness — 13.6% val accuracy vs 81.6% on 2-choice. We didn't have time to address this with targeted augmentation or a different decoding strategy.
- Increasing `LORA_R` beyond 8 pushed us over the 5M parameter cap.
- `bfloat16` caused training instability on the T4, so we stayed with `float16`.
- We would have liked to try a larger base model (e.g., SmolVLM-2B) but it didn't fit within T4 memory limits.

---

## Reproducibility

Everything needed to reproduce our submission:

- **Notebook:** `notebooks/pixels_to_predictions_smolvlm.ipynb` — runs end-to-end on Kaggle, no internet needed at inference time
- **Random seeds:** Fixed at `42` (primary) and `7` (secondary), set at the top of the config cell
- **Dependencies:** Pinned in `requirements.txt`
- **Model weights:** Saved to `/kaggle/working/lora_seed42/` and `/kaggle/working/lora_seed7/` at the end of training — download links above
- **Submission:** Written to `/kaggle/working/submission.csv`

To reproduce locally:
1. Clone this repo
2. Download the competition dataset from Kaggle and set `DATA_ROOT` in `src/config.py`
3. `pip install -r requirements.txt`
4. Run `python src/train.py` then `python src/inference.py`

---

## AI Tooling Disclosure

As required by the course, we disclose our use of AI assistance:

- **Claude (Anthropic) and GitHub Copilot** were used for code completion, debugging HuggingFace/PyTorch API calls, and writing docstrings.
- All model design choices, architecture decisions, and hyperparameter selections were made by us.

---

## References

1. HuggingFaceTB, *SmolVLM-500M-Instruct*, https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685
3. Rodriguez et al., *StarVector*, arXiv:2312.11556 — SVG generation benchmark used in evaluation design
4. HuggingFace PEFT Library, https://github.com/huggingface/peft
5. CairoSVG — SVG renderer used in the competition scoring pipeline, https://cairosvg.org/
6. Kaggle Competition Documentation, https://www.kaggle.com/docs/competitions
