from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import SFTConfig, SFTTrainer

from .config import LoRAConfig, ModelConfig, TrainingConfig, WandbConfig


class _RunIdCaptureCallback(TrainerCallback):
    """Capture the W&B run ID opened by HuggingFace's WandbCallback."""

    def __init__(self) -> None:
        self.run_id: Optional[str] = None

    def on_train_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        try:
            import wandb
            if wandb.run is not None:
                self.run_id = wandb.run.id
        except ImportError:
            pass


class LoRATrainer:
    """Wraps SFTTrainer with report_to='wandb' for automatic metric logging."""

    def __init__(
        self,
        model,
        tokenizer,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        wandb_config: WandbConfig,
        run_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.model_config = model_config
        self.lora_config = lora_config
        self.wandb_config = wandb_config
        self.run_name = run_name
        self._trainer: Optional[SFTTrainer] = None
        self._train_result = None
        self._run_id_callback = _RunIdCaptureCallback()

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
            run_name=self.run_name or self.wandb_config.run_name,
            report_to="wandb",
            gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="text",
            packing=False,
            max_seq_length=self.model_config.max_seq_length,
        )

    @property
    def wandb_run_id(self) -> Optional[str]:
        """Return the W&B run ID created during training, if available."""
        return self._run_id_callback.run_id

    def train(self, train_dataset: Dataset) -> Dict[str, Any]:
        """
        Run supervised fine-tuning.

        Training metrics and hyperparameters are logged to W&B automatically
        via HuggingFace's built-in WandbCallback (report_to='wandb').
        """
        sft_config = self._build_sft_config()
        self._trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
            callbacks=[self._run_id_callback],
        )
        cfg = self.training_config
        print("Starting LoRA fine-tuning …")
        print(f"  Epochs:         {cfg.num_train_epochs}")
        print(f"  Effective batch:{cfg.effective_batch_size}  "
              f"({cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps})")
        print(f"  Learning rate:  {cfg.learning_rate}")
        print(f"  W&B logging:    report_to=wandb (project: {self.wandb_config.project})")

        self._train_result = self._trainer.train()
        metrics = self._train_result.metrics
        print(f"\nTraining complete.")
        print(f"  Steps:      {self._train_result.global_step}")
        print(f"  Runtime:    {metrics['train_runtime']:.0f}s")
        print(f"  Final loss: {metrics['train_loss']:.4f}")
        if self.wandb_run_id:
            print(f"  W&B run:    {self.wandb_run_id}")
        return metrics

    def get_log_history(self) -> List[Dict[str, Any]]:
        """Return the full per-step log history from the trainer state."""
        return self._trainer.state.log_history if self._trainer else []

    def get_step_losses(self) -> List[Dict[str, Any]]:
        """Return only the entries that contain a step-level loss value."""
        return [e for e in self.get_log_history() if "loss" in e]
