import os
from typing import Any, Dict, Optional

import wandb

from .config import DataConfig, LoRAConfig, ModelConfig, PromptConfig, TrainingConfig, WandbConfig


class WandbLogger:
    """
    Centralises all Weights & Biases interactions: run lifecycle, config logging,
    metric streaming, artifact uploads, dataset tracking, and model versioning.
    """

    def __init__(self, config: WandbConfig) -> None:
        self.config = config
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self._configure_auth()

    def _configure_auth(self) -> None:
        """
        Push W&B credentials and server URL into os.environ so the W&B client
        connects to the correct instance. Credentials stored by `wandb login`
        in ~/.netrc are used automatically when api_key is not set here.
        base_url is only applied when explicitly set (self-hosted server);
        when None the W&B client defaults to https://api.wandb.ai (cloud).
        """
        if self.config.api_key:
            os.environ.setdefault("WANDB_API_KEY", self.config.api_key)
        if self.config.base_url:
            os.environ.setdefault("WANDB_BASE_URL", self.config.base_url)
        server = self.config.base_url or "https://api.wandb.ai (cloud)"
        print(f"W&B configured — server: {server}")

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> Optional[str]:
        """Return the active run ID, or None if no run is open."""
        return self._run.id if self._run else None

    def start_run(self, run_name: Optional[str] = None, group: Optional[str] = None) -> None:
        """
        Begin a new W&B training run.

        Both the training and eval runs receive the same group name so they
        are displayed together under one group in the W&B UI.
        """
        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=run_name or self.config.run_name,
            group=group,
            job_type="training",
            tags=["lora", "sql", "qwen2.5"],
            reinit=True,
        )
        print(f"W&B run started — ID: {self._run.id}  "
              f"URL: {self._run.get_url()}")

    def start_eval_run(
        self,
        training_run_id: str,
        group: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Open a separate evaluation run in the same W&B project, linked to the
        training run via group membership and a config entry.
        """
        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=run_name or f"eval_{training_run_id}",
            group=group,
            job_type="evaluation",
            tags=["evaluation", "lora", "sql"],
            config={"training_run_id": training_run_id},
            reinit=True,
        )
        print(f"W&B eval run started — ID: {self._run.id}  "
              f"URL: {self._run.get_url()}")

    def finish_run(self) -> None:
        """Finalise the current run and print a direct link to the W&B UI."""
        if self._run:
            url = self._run.get_url()
            self._run.finish()
            print(f"W&B run finished.\n  View at: {url}")
            self._run = None

    # ------------------------------------------------------------------
    # Config / hyperparameter logging
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
    # Artifact / file logging
    # ------------------------------------------------------------------

    def log_artifact_file(self, local_path: str, artifact_subdir: str = "plots") -> None:
        """
        Upload a local file to the current run.

        PNG/JPG images are also sent as rich media panels (wandb.Image) so they
        appear in the W&B Media tab in addition to the Artifacts tab.
        """
        if not os.path.exists(local_path):
            print(f"Warning: artifact not found, skipping: {local_path}")
            return

        ext = os.path.splitext(local_path)[1].lower()
        filename = os.path.basename(local_path)

        if ext in (".png", ".jpg", ".jpeg"):
            wandb.log({f"{artifact_subdir}/{filename}": wandb.Image(local_path)})

        artifact = wandb.Artifact(
            name=filename.replace(".", "_"),
            type=artifact_subdir,
        )
        artifact.add_file(local_path)
        self._run.log_artifact(artifact)

    # ------------------------------------------------------------------
    # Prompt versioning
    # ------------------------------------------------------------------

    def log_prompt_artifact(self, prompt_config: PromptConfig) -> None:
        """
        Version the system prompt in W&B as a prompt artifact.

        Each call creates a new artifact version so prompt changes are tracked
        over time, similar to MLflow's Prompt Registry.
        """
        artifact = wandb.Artifact(
            name=self.config.prompt_artifact_name,
            type="prompt",
            description=prompt_config.commit_message,
            metadata={"commit_message": prompt_config.commit_message},
        )
        with artifact.new_file("system_prompt.txt", mode="w") as fh:
            fh.write(prompt_config.text)
        self._run.log_artifact(artifact, aliases=["latest"])
        print(f"System prompt logged as W&B artifact: '{self.config.prompt_artifact_name}'")

    def load_system_prompt(self, fallback_text: str) -> str:
        """
        Load the latest system prompt from W&B artifacts.

        On the very first run the artifact does not exist yet, so this falls
        back to fallback_text (i.e. PromptConfig.text) and prints a notice.
        On all subsequent runs the versioned prompt from W&B is used, allowing
        prompt edits to be made via the W&B UI without touching source code.
        """
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
            print(
                f"System prompt loaded from W&B artifact: "
                f"'{self.config.prompt_artifact_name}:latest' (v{artifact.version})"
            )
            return text
        except Exception as exc:
            print(
                f"Could not load prompt from W&B ({exc}). "
                f"Using PromptConfig default."
            )
            return fallback_text

    # ------------------------------------------------------------------
    # Dataset logging
    # ------------------------------------------------------------------

    def log_dataset(
        self,
        train_data,
        test_data,
        data_config: DataConfig,
    ) -> None:
        """
        Log train and test splits as a versioned W&B dataset artifact.

        A wandb.Table of up to 10 sample rows is embedded so the data is
        immediately previewable in the W&B Artifacts UI without downloading.
        """
        artifact = wandb.Artifact(
            name=self.config.dataset_artifact_name,
            type="dataset",
            description=f"Text-to-SQL dataset — {data_config.dataset_name}",
            metadata={
                "source": data_config.dataset_name,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "num_train_requested": data_config.num_train,
                "num_test_requested": data_config.num_test,
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
        print(
            f"Dataset logged as W&B artifact '{self.config.dataset_artifact_name}': "
            f"{len(train_data):,} train / {len(test_data):,} test."
        )

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------

    def log_final_model(
        self,
        model_loader,
        tokenizer,
        tmp_dir: str = "/tmp/lora_adapter",
    ) -> None:
        """
        Save the LoRA adapter weights locally, upload them as a versioned W&B
        model artifact, and block until the upload is confirmed complete.

        Calling .wait() on the returned artifact handle ensures the upload
        finishes before finish_run() is called — without it the run can close
        while the artifact is still queued, causing it to appear missing.
        """
        import shutil
        import traceback

        # Always start from a clean temp directory so stale files from a
        # previous run cannot pollute the artifact.
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            model_loader.save_adapter(tmp_dir)

            saved_files = os.listdir(tmp_dir)
            if not saved_files:
                raise RuntimeError(
                    f"save_adapter() produced no files in {tmp_dir}."
                )
            print(f"Adapter files saved: {saved_files}")

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
            # Block until W&B confirms the upload is complete so that a
            # subsequent finish_run() does not close the run prematurely.
            logged.wait()
            print(
                f"Final model logged as W&B artifact "
                f"'{self.config.model_artifact_name}' (v{logged.version})."
            )
        except Exception as exc:
            print(f"ERROR: could not log model artifact to W&B — {exc}")
            traceback.print_exc()
