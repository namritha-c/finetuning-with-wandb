import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PromptConfig:
    text: str = (
        "You are a SQL assistant. Given a database schema and a question, "
        "output ONLY the SQL query that answers the question. "
        "Do not include any explanation, markdown formatting, or extra text."
    )
    commit_message: str = "Registered via training run"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 1024
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"


@dataclass
class DataConfig:
    dataset_name: str = "b-mc2/sql-create-context"
    target_column: str = "answer"
    num_train: int = 500
    num_test: int = 50
    seed: int = 42


@dataclass
class TrainingConfig:
    output_dir: str = "./lora_sql_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "no"
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass
class WandbConfig:
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_BASE_URL")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_API_KEY")
    )
    entity: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_ENTITY")
    )

    project: str = "SQL_Finetuning_HF"
    description: str = "Fine tuning qwen model for SQL query prediction (HF report_to=wandb)."
    run_name: Optional[str] = "sql_finetuning"

    prompt_artifact_name: str = "sql-assistant-system-prompt"
    model_artifact_name: str = "sql-model"
    dataset_artifact_name: str = "sql-create-context"
