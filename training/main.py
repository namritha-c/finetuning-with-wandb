"""
Entry point for LoRA fine-tuning with Weights & Biases experiment tracking.

Usage:
    python -m training.main          (recommended, from project root)
    python training/main.py          (also works, from project root)
"""

import os
import random
import sys
import time

import torch

# Ensure the project root (parent of this package) is on sys.path so that
# `from training import ...` resolves correctly when the script is run
# directly as `python training/main.py` instead of `python -m training.main`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training import (
    DataConfig,
    DatasetLoader,
    LoRAConfig,
    LoRATrainer,
    ModelConfig,
    ModelEvaluator,
    ModelLoader,
    PromptConfig,
    TrainingConfig,
    Visualizer,
    WandbConfig,
    WandbLogger,
)


def _setup_environment(seed: int = 42) -> None:
    # CUDA_VISIBLE_DEVICES and TOKENIZERS_PARALLELISM are already in os.environ
    # via load_dotenv() called at config import time; setdefault is a safe fallback.
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Configs
    # ------------------------------------------------------------------
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()
    wandb_config = WandbConfig()
    prompt_config = PromptConfig()

    _setup_environment(seed=data_config.seed)

    # Shared group name ties the training and eval runs together in the W&B UI.
    run_group = f"lora_sql_{int(time.time())}"

    # All local output files (plots, etc.) go here — keeps the project root clean.
    plots_dir = os.path.join(_PROJECT_ROOT, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. W&B — open training run and log all configs upfront
    # ------------------------------------------------------------------
    wandb_logger = WandbLogger(wandb_config)
    wandb_logger.start_run(
        run_name=f"{run_group}_train",
        group=run_group,
    )

    wandb_logger.log_model_config(model_config)
    wandb_logger.log_lora_config(lora_config)
    wandb_logger.log_training_config(training_config)
    wandb_logger.log_data_config(data_config)

    # ------------------------------------------------------------------
    # 3. System prompt — log to W&B artifact registry, then load it back
    # ------------------------------------------------------------------
    print("\n--- Logging system prompt to W&B ---")
    wandb_logger.log_prompt_artifact(prompt_config)
    # load_system_prompt falls back to prompt_config.text on the first run
    # (before the artifact upload has finalised); on subsequent runs it loads
    # the versioned prompt from W&B so edits made via the UI take effect.
    system_prompt = wandb_logger.load_system_prompt(fallback_text=prompt_config.text)

    # ------------------------------------------------------------------
    # 4. Dataset — load and log to W&B with sample preview table
    # ------------------------------------------------------------------
    print("\n--- Loading dataset ---")
    data_loader = DatasetLoader(data_config, system_prompt=system_prompt)
    data_loader.load()

    wandb_logger.log_dataset(data_loader.train_data, data_loader.test_data, data_config)

    # ------------------------------------------------------------------
    # 5. Base model — load and evaluate BEFORE fine-tuning
    # ------------------------------------------------------------------
    print("\n--- Loading base model ---")
    model_loader = ModelLoader(model_config, lora_config)
    model, tokenizer = model_loader.load_base_model()

    print("\n--- Evaluating base model ---")
    base_evaluator = ModelEvaluator(model, tokenizer, data_loader)
    base_eval = base_evaluator.evaluate(data_loader.test_data, label="Base Model")
    base_evaluator.print_sample_outputs(base_eval, n=3, label="Base Model")

    # ------------------------------------------------------------------
    # 6. Apply LoRA and prepare training data
    # ------------------------------------------------------------------
    print("\n--- Applying LoRA adapters ---")
    model_loader.apply_lora(seed=training_config.seed)
    total_params, trainable_params = model_loader.get_parameter_counts()

    print("\n--- Preparing training dataset ---")
    train_dataset = data_loader.prepare_training_dataset(tokenizer)
    print(f"Training dataset ready: {len(train_dataset):,} examples")

    # ------------------------------------------------------------------
    # 7. Train — HF WandbCallback streams metrics live to W&B automatically
    # ------------------------------------------------------------------
    print("\n--- Training ---")
    lora_trainer = LoRATrainer(
        model_loader.model,
        tokenizer,
        training_config,
        model_config,
        lora_config,
        wandb_config,
        run_name=f"{run_group}_train",
    )
    train_metrics = lora_trainer.train(train_dataset)

    # ------------------------------------------------------------------
    # 8. Training loss plot
    # ------------------------------------------------------------------
    loss_plot_path = os.path.join(plots_dir, "training_loss.png")
    Visualizer.plot_training_loss(lora_trainer.get_log_history(), save_path=loss_plot_path)
    wandb_logger.log_artifact_file(loss_plot_path, artifact_subdir="plots")

    # ------------------------------------------------------------------
    # 9. Log final model to W&B as a versioned model artifact
    # ------------------------------------------------------------------
    print("\n--- Logging final model to W&B ---")
    wandb_logger.log_final_model(model_loader, tokenizer)

    # Capture the training run ID before closing; the eval run references it.
    training_run_id = wandb_logger.run_id
    wandb_logger.finish_run()

    # ------------------------------------------------------------------
    # 10. Evaluation run — separate W&B run, same group, job_type=evaluation
    # ------------------------------------------------------------------
    wandb_logger.start_eval_run(
        training_run_id=training_run_id,
        group=run_group,
        run_name=f"{run_group}_eval",
    )

    print("\n--- Evaluating fine-tuned model ---")
    ft_evaluator = ModelEvaluator(model_loader.model, tokenizer, data_loader)
    finetuned_eval = ft_evaluator.evaluate(data_loader.test_data, label="Fine-Tuned Model")
    ft_evaluator.print_sample_outputs(finetuned_eval, n=3, label="Fine-Tuned Model")

    # ------------------------------------------------------------------
    # 11. Log evaluation metrics and visualisations
    # ------------------------------------------------------------------
    wandb_logger.log_evaluation_metrics(
        base_eval, finetuned_eval, total_params, trainable_params
    )

    acc_plot_path = os.path.join(plots_dir, "accuracy_comparison.png")
    Visualizer.plot_accuracy_comparison(
        base_eval["accuracy"], finetuned_eval["accuracy"], save_path=acc_plot_path
    )
    wandb_logger.log_artifact_file(acc_plot_path, artifact_subdir="plots")

    summary_plot_path = os.path.join(plots_dir, "summary.png")
    Visualizer.plot_summary(
        base_acc=base_eval["accuracy"],
        ft_acc=finetuned_eval["accuracy"],
        total_params=total_params,
        trainable_params=trainable_params,
        train_runtime_seconds=train_metrics.get("train_runtime", 0.0),
        save_path=summary_plot_path,
    )
    wandb_logger.log_artifact_file(summary_plot_path, artifact_subdir="plots")

    # ------------------------------------------------------------------
    # 12. Print final summary and close eval run
    # ------------------------------------------------------------------
    base_acc = base_eval["accuracy"]
    ft_acc = finetuned_eval["accuracy"]
    print("\n" + "=" * 55)
    print(f"{'METRIC':<28} {'BASE':>10} {'FINE-TUNED':>12}")
    print("=" * 55)
    print(f"{'Exact-Match Accuracy':<28} {base_acc:>9.1f}% {ft_acc:>11.1f}%")
    print(f"{'Correct / Total':<28} {base_eval['correct']:>4}/{base_eval['total']:<5} "
          f"{finetuned_eval['correct']:>4}/{finetuned_eval['total']:<5}")
    print("=" * 55)
    print(f"\nAccuracy improvement : +{ft_acc - base_acc:.1f} pp")
    print(f"Parameters trained   : {trainable_params:,} / {total_params:,} "
          f"({trainable_params / total_params * 100:.2f}%)")
    print(f"Training time        : {train_metrics.get('train_runtime', 0):.0f}s")

    wandb_logger.finish_run()


if __name__ == "__main__":
    main()
