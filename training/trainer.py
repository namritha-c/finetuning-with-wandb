from typing import Any, Dict, List, Optional

import torch
import wandb
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig, SFTTrainer

from .config import ModelConfig, TrainingConfig


class WandbStepCallback(TrainerCallback):
    """
    Streams per-step training metrics to W&B at every logging interval.

    This is equivalent to the real-time metric logging that MLflow achieves
    with its step callback, ensuring the loss curve appears live in the W&B
    UI during training rather than only after the run completes.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if wandb.run is None or not logs:
            return

        log_dict: Dict[str, Any] = {}
        if "loss" in logs:
            log_dict["train/step_loss"] = logs["loss"]
        if "grad_norm" in logs:
            log_dict["train/step_grad_norm"] = logs["grad_norm"]
        if "learning_rate" in logs:
            log_dict["train/step_learning_rate"] = logs["learning_rate"]
        if "epoch" in logs:
            log_dict["train/epoch"] = logs["epoch"]

        if log_dict:
            wandb.log(log_dict, step=state.global_step)


class LoRATrainer:
    """Wraps SFTTrainer with the W&B step callback for real-time metric visibility."""

    def __init__(
        self,
        model,
        tokenizer,
        training_config: TrainingConfig,
        model_config: ModelConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.model_config = model_config
        self._trainer: Optional[SFTTrainer] = None
        self._train_result = None

    def _build_sft_config(self) -> SFTConfig:
        use_bf16 = torch.cuda.is_bf16_supported()
        return SFTConfig(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_strategy=self.training_config.save_strategy,
            seed=self.training_config.seed,
            fp16=not use_bf16,
            bf16=use_bf16,
            # Disable built-in W&B / MLflow reporting — WandbStepCallback owns logging.
            report_to="none",
            dataset_text_field="text",
            packing=False,
            max_seq_length=self.model_config.max_seq_length,
        )

    def train(self, train_dataset: Dataset) -> Dict[str, Any]:
        """Run supervised fine-tuning and return the training metrics dict."""
        sft_config = self._build_sft_config()
        self._trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
            callbacks=[WandbStepCallback()],
        )
        self._train_result = self._trainer.train()
        return self._train_result.metrics

    def get_log_history(self) -> List[Dict[str, Any]]:
        """Return the full per-step log history from the trainer state."""
        return self._trainer.state.log_history if self._trainer else []

    def get_step_losses(self) -> List[Dict[str, Any]]:
        """Return only the entries that contain a step-level loss value."""
        return [e for e in self.get_log_history() if "loss" in e]
