# LoRA Fine-Tuning with Weights & Biases Experiment Tracking

QLoRA fine-tuning of **Qwen2.5-3B-Instruct** on a Text-to-SQL task using [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl), with full experiment tracking via **Weights & Biases**.

---

## What this does

1. Evaluates the **base model** accuracy on a held-out SQL test set
2. Fine-tunes with **LoRA** (rank 16, ~1.6 % of parameters, ~23 min on a single GPU)
3. Re-evaluates the **fine-tuned model** on the same test set
4. Logs everything to W&B — hyperparameters, per-step metrics, evaluation results, plots, dataset preview, and the LoRA adapter weights as versioned artifacts

| | Base Model | Fine-Tuned |
|---|---|---|
| Task | Text-to-SQL | Text-to-SQL |
| Dataset | `b-mc2/sql-create-context` | same |
| Exact-Match Accuracy | ~4.8 % | ~75.8 % |
| Trainable params | 3 B (frozen) | ~30 M (LoRA only) |

---

## Project layout

```
training_with_wandb/
├── .env.template            # environment variable reference
├── .gitignore
├── requirements.txt
└── training/
    ├── __init__.py          # package re-exports
    ├── config.py            # dataclass configs (Model, LoRA, Data, Training, Wandb)
    ├── data_loader.py       # DatasetLoader  — download, split, prompt formatting
    ├── model_loader.py      # ModelLoader    — 4-bit load + LoRA adapter application
    ├── evaluator.py         # ModelEvaluator — exact-match SQL evaluation loop
    ├── trainer.py           # LoRATrainer    — SFTTrainer wrapper + WandbStepCallback
    ├── visualizer.py        # Visualizer     — loss curve, accuracy bars, summary chart
    ├── wandb_logger.py      # WandbLogger    — all W&B interactions
    └── main.py              # entry point   — orchestrates every step
```

---

## Requirements

- Python 3.11+
- CUDA-capable GPU with at least **16 GB VRAM** (tested on a single A100/H100)
- A [Weights & Biases](https://wandb.ai) account (free tier is sufficient)

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Log in to W&B (one-time; stores credentials in ~/.netrc)
wandb login
# For a self-hosted W&B server:
# wandb login --host=http://<host>:8080
```

> **Note:** `unsloth` bundles a custom CUDA kernel build. If the wheel for your CUDA version is not available on PyPI, follow the [Unsloth installation guide](https://github.com/unslothai/unsloth#installation).

---

## Configuration

### Environment variables

Copy `.env.template` to `.env` and fill in your values:

```bash
cp .env.template .env
```

```
WANDB_ENTITY=your_username_or_team   # shown after wandb login
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
```

`WANDB_API_KEY` and `WANDB_BASE_URL` are **not required** when using the hosted cloud service after `wandb login`.

### Hyperparameters

All hyperparameters live in `training/config.py` as plain dataclasses — no YAML or CLI flags required. Edit the defaults directly before running:

| Dataclass | Key fields |
|---|---|
| `ModelConfig` | `name`, `max_seq_length`, `load_in_4bit` |
| `LoRAConfig` | `r`, `lora_alpha`, `lora_dropout`, `target_modules` |
| `DataConfig` | `dataset_name`, `num_train`, `num_test`, `seed` |
| `TrainingConfig` | `num_train_epochs`, `learning_rate`, `per_device_train_batch_size` |
| `WandbConfig` | `project`, `entity`, `prompt_artifact_name`, `model_artifact_name` |

---

## Running

```bash
# from the training_with_wandb/ directory
python -m training.main
```

The script will:

1. Authenticate with W&B using credentials stored by `wandb login`
2. Create a **run group** (e.g. `lora_sql_1748860000`) shared between the training and evaluation runs
3. Open the **training run** and log all hyperparameters to the W&B Config panel
4. Version the system prompt as a W&B artifact
5. Load the dataset and log it as a W&B artifact with a 10-row preview table
6. Evaluate the base model, apply LoRA, and fine-tune (per-step metrics stream live)
7. Log the LoRA adapter weights as a versioned **model artifact**
8. Close the training run and open the linked **evaluation run**
9. Evaluate the fine-tuned model, log accuracy metrics and three plots
10. Print a final accuracy comparison table and the W&B run URLs

---

## W&B tracking

Your runs are visible at:
`https://wandb.ai/<entity>/LoRA_SQL_Finetuning`

### Run structure

Each pipeline execution produces **two runs** linked under the same group:

| Run | `job_type` | Contents |
|---|---|---|
| `lora_sql_<ts>_train` | `training` | hyperparams, step metrics, loss plot, model artifact |
| `lora_sql_<ts>_eval` | `evaluation` | accuracy metrics, comparison plots, `training_run_id` config |

Grouping both runs lets you compare training and evaluation results side-by-side in the W&B UI under **Runs → Group**.

### Hyperparameters (W&B Config)

| Prefix | Examples |
|---|---|
| `model_*` | `model_name`, `model_max_seq_length`, `model_load_in_4bit` |
| `lora_*` | `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules` |
| `training_*` | `training_num_epochs`, `training_learning_rate`, `training_effective_batch_size` |
| `data_*` | `data_dataset_name`, `data_num_train`, `data_num_test` |

### Metrics (W&B Charts)

| Metric | Description |
|---|---|
| `train/step_loss` | Per-step training loss — streams live during training |
| `train/step_grad_norm` | Per-step gradient norm |
| `train/step_learning_rate` | Learning rate schedule |
| `train/loss_final` | Final training loss |
| `train/runtime_seconds` | Wall-clock training time |
| `eval/base_accuracy` | Exact-match accuracy before fine-tuning |
| `eval/finetuned_accuracy` | Exact-match accuracy after fine-tuning |
| `eval/accuracy_improvement` | Absolute percentage-point gain |
| `model/trainable_parameters` | Number of LoRA trainable parameters |
| `model/trainable_percentage` | Trainable % of total parameters |

### Artifacts

| Name | Type | Contents |
|---|---|---|
| `sql-assistant-system-prompt` | `prompt` | `system_prompt.txt` — versioned on every run |
| `sql-create-context` | `dataset` | metadata + 10-row `train_samples` table |
| `lora-sql-model` | `model` | adapter weights (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files) |
| `training_loss_png` | `plots` | loss curve PNG |
| `accuracy_comparison_png` | `plots` | before vs after bar chart |
| `summary_png` | `plots` | 3-panel: accuracy + parameter pie + training time |

Plots are also logged as **W&B Images** and appear in the **Media** tab of the run.

---

## Loading the saved adapter

Download from the W&B Artifacts UI or via the SDK:

```python
import wandb

run = wandb.init(project="LoRA_SQL_Finetuning", entity="<your-entity>")
artifact = run.use_artifact("lora-sql-model:latest", type="model")
adapter_dir = artifact.download()   # returns local path
```

Then load for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct", load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
```

---

## Self-hosted W&B server

If you want to run W&B on your own machine instead of using the cloud service, Docker is required:

```bash
# Option A — W&B CLI (recommended)
pip install wandb --upgrade
wandb server start -p 8080

# Option B — plain Docker
docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
```

Then update `.env`:

```
WANDB_BASE_URL=http://<host>:8080
WANDB_API_KEY=<key-from-settings-page>
```

---

## Dataset

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — ~78 K examples from WikiSQL and Spider, each containing:

- `question` — natural language question
- `context` — `CREATE TABLE` schema
- `answer` — expected SQL query

By default, 5 000 examples are used for training and 500 for evaluation.
