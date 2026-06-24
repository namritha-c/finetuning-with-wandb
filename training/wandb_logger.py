import os
import shutil
import time
import traceback
from typing import Any, Dict, List, Optional

import wandb

from .config import DataConfig, LoRAConfig, ModelConfig, PromptConfig, TrainingConfig, WandbConfig
from .model_loader import ModelLoader


class WandbLogger:
    """
    Centralises all W&B interactions: run lifecycle, config/metric logging,
    artifact uploads, dataset tracking, and model versioning.

    Training step metrics are handled automatically by HuggingFace's
    WandbCallback (report_to="wandb"). This class manages the run lifecycle,
    config logging, post-training artifacts, and evaluation runs.
    """

    def __init__(self, config: WandbConfig) -> None:
        self.config = config
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self._configure_auth()

    def _configure_auth(self) -> None:
        if self.config.api_key:
            os.environ.setdefault("WANDB_API_KEY", self.config.api_key)
        if self.config.base_url:
            os.environ.setdefault("WANDB_BASE_URL", self.config.base_url)
        os.environ["WANDB_PROJECT"] = self.config.project
        server = self.config.base_url or "https://api.wandb.ai (cloud)"
        print(f"W&B configured — server: {server}")

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> Optional[str]:
        return self._run.id if self._run else None

    def start_run(self) -> None:
        run_name = f"{self.config.run_name}_{int(time.time())}"
        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=run_name,
            job_type="training",
            tags=["lora", "sql", "qwen2.5"],
            notes=self.config.description,
            reinit=True,
        )
        print(f"W&B run started — ID: {self._run.id}  URL: {self._run.get_url()}")

    def start_eval_run(self, training_run_id: str, run_name: Optional[str] = None) -> None:
        name = run_name or f"eval_{training_run_id[:8]}"
        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=name,
            job_type="evaluation",
            tags=["evaluation", "lora", "sql"],
            config={"training_run_id": training_run_id},
            reinit=True,
        )
        print(f"W&B eval run started — ID: {self._run.id}  URL: {self._run.get_url()}")

    def end_run(self) -> None:
        if self._run:
            url = self._run.get_url()
            self._run.finish()
            print(f"W&B run finished.\n  View at: {url}")
            self._run = None

    # ------------------------------------------------------------------
    # Config / parameter logging
    # ------------------------------------------------------------------

    def log_model_config(self, config: ModelConfig) -> None:
        wandb.config.update({
            "model_name": config.name,
            "model_max_seq_length": config.max_seq_length,
            "model_load_in_4bit": config.load_in_4bit,
        })

    def log_lora_config(self, config: LoRAConfig) -> None:
        wandb.config.update({
            "lora_r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": ",".join(config.target_modules),
            "lora_bias": config.bias,
        })

    def log_training_config(self, config: TrainingConfig) -> None:
        wandb.config.update({
            "training_num_epochs": config.num_train_epochs,
            "training_per_device_batch_size": config.per_device_train_batch_size,
            "training_gradient_accumulation_steps": config.gradient_accumulation_steps,
            "training_effective_batch_size": config.effective_batch_size,
            "training_learning_rate": config.learning_rate,
            "training_lr_scheduler_type": config.lr_scheduler_type,
            "training_warmup_ratio": config.warmup_ratio,
            "training_save_strategy": config.save_strategy,
            "training_seed": config.seed,
        })

    def log_data_config(self, config: DataConfig) -> None:
        wandb.config.update({
            "data_dataset_name": config.dataset_name,
            "data_num_train": config.num_train,
            "data_num_test": config.num_test,
            "data_seed": config.seed,
        })

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_training_metrics(self, metrics: Dict[str, Any]) -> None:
        wandb.log({
            "train/loss_final": metrics.get("train_loss", 0.0),
            "train/runtime_seconds": metrics.get("train_runtime", 0.0),
            "train/samples_per_second": metrics.get("train_samples_per_second", 0.0),
            "train/steps_per_second": metrics.get("train_steps_per_second", 0.0),
        })

    def log_evaluation_metrics(
        self,
        base_eval: Dict[str, Any],
        finetuned_eval: Dict[str, Any],
        total_params: int,
        trainable_params: int,
    ) -> None:
        wandb.log({
            "eval/base_accuracy": base_eval["accuracy"],
            "eval/base_correct": base_eval["correct"],
            "eval/base_elapsed_seconds": base_eval["elapsed_seconds"],
            "eval/finetuned_accuracy": finetuned_eval["accuracy"],
            "eval/finetuned_correct": finetuned_eval["correct"],
            "eval/finetuned_elapsed_seconds": finetuned_eval["elapsed_seconds"],
            "eval/accuracy_improvement": finetuned_eval["accuracy"] - base_eval["accuracy"],
            "model/total_parameters": float(total_params),
            "model/trainable_parameters": float(trainable_params),
            "model/trainable_percentage": trainable_params / total_params * 100,
        })

    # ------------------------------------------------------------------
    # Artifact logging
    # ------------------------------------------------------------------

    def log_artifact(self, local_path: str, artifact_subdir: str = "") -> None:
        if not os.path.exists(local_path):
            print(f"Warning: artifact not found, skipping: {local_path}")
            return

        ext = os.path.splitext(local_path)[1].lower()
        filename = os.path.basename(local_path)

        if ext in (".png", ".jpg", ".jpeg"):
            wandb.log({f"{artifact_subdir}/{filename}": wandb.Image(local_path)})

        artifact = wandb.Artifact(
            name=filename.replace(".", "_"),
            type=artifact_subdir or "file",
        )
        artifact.add_file(local_path)
        self._run.log_artifact(artifact)

    # ------------------------------------------------------------------
    # Prompt versioning
    # ------------------------------------------------------------------

    def register_system_prompt(self, prompt_config: PromptConfig) -> None:
        artifact = wandb.Artifact(
            name=self.config.prompt_artifact_name,
            type="prompt",
            description=prompt_config.commit_message,
        )
        with artifact.new_file("system_prompt.txt", mode="w") as fh:
            fh.write(prompt_config.text)
        self._run.log_artifact(artifact, aliases=["latest"])
        print(f"System prompt logged as W&B artifact: '{self.config.prompt_artifact_name}'")

    def load_system_prompt_from_registry(self, prompt_config: PromptConfig) -> str:
        try:
            entity = self.config.entity or self._run.entity
            artifact_path = (
                f"{entity}/{self.config.project}/"
                f"{self.config.prompt_artifact_name}:latest"
            )
            api = wandb.Api(api_key=self.config.api_key)
            artifact = api.artifact(artifact_path, type="prompt")
            artifact_dir = artifact.download()
            prompt_file = os.path.join(artifact_dir, "system_prompt.txt")
            with open(prompt_file) as fh:
                text = fh.read().strip()
            print(f"System prompt loaded from W&B artifact: "
                  f"'{self.config.prompt_artifact_name}:latest' (v{artifact.version})")
            return text
        except Exception as exc:
            print(f"Could not load prompt from W&B ({exc}). Using PromptConfig default.")
            return prompt_config.text

    # ------------------------------------------------------------------
    # Dataset logging
    # ------------------------------------------------------------------

    def log_dataset(self, train_data, test_data, data_config: DataConfig) -> None:
        artifact = wandb.Artifact(
            name=self.config.dataset_artifact_name,
            type="dataset",
            description=f"Text-to-SQL dataset — {data_config.dataset_name}",
            metadata={
                "source": data_config.dataset_name,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "seed": data_config.seed,
            },
        )
        sample_table = wandb.Table(columns=["question", "context", "answer"])
        for i in range(min(10, len(train_data))):
            sample_table.add_data(
                train_data[i]["question"],
                train_data[i]["context"][:300],
                train_data[i]["answer"],
            )
        artifact.add(sample_table, "train_samples")
        self._run.log_artifact(artifact)
        print(f"Dataset logged as W&B artifact '{self.config.dataset_artifact_name}': "
              f"{len(train_data):,} train / {len(test_data):,} test.")

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------

    def log_final_model(self, model_loader: ModelLoader, tokenizer) -> None:
        tmp_dir = "/tmp/lora_adapter"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            model_loader.save_adapter(tmp_dir)
            artifact = wandb.Artifact(
                name=self.config.model_artifact_name,
                type="model",
                description="LoRA adapter fine-tuned for Text-to-SQL",
                metadata={
                    "base_model": model_loader.model_config.name,
                    "adapter_type": "lora",
                    "task": "text-to-sql",
                    "framework": "transformers+peft",
                },
            )
            artifact.add_dir(tmp_dir)
            logged = self._run.log_artifact(artifact)
            logged.wait()
            print(f"Model logged as W&B artifact '{self.config.model_artifact_name}'.")
        except Exception as exc:
            print(f"ERROR: could not log model artifact to W&B — {exc}")
            traceback.print_exc()

    def log_step_losses(self, log_history: List[Dict[str, Any]]) -> None:
        for entry in log_history:
            if "loss" in entry:
                wandb.log({"step_train_loss": entry["loss"]}, step=entry["step"])
