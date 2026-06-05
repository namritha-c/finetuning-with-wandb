from typing import Optional, Tuple

import torch
from unsloth import FastLanguageModel

from .config import LoRAConfig, ModelConfig


class ModelLoader:
    """Responsible for loading the base model and applying LoRA adapters."""

    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig) -> None:
        self.model_config = model_config
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None

    def load_base_model(self) -> Tuple:
        """Load the 4-bit quantised base model and its tokenizer."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.name,
            max_seq_length=self.model_config.max_seq_length,
            load_in_4bit=self.model_config.load_in_4bit,
            dtype=self.model_config.dtype,
        )
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Base model loaded: {self.model_config.name}")
        print(f"  Total parameters: {total:,}")
        return self.model, self.tokenizer

    def apply_lora(self, seed: int = 42) -> None:
        """Wrap the base model with PEFT LoRA adapters (in-place)."""
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            random_state=seed,
        )
        total, trainable = self.get_parameter_counts()
        print(f"LoRA adapters applied.")
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.4f}%)")

    def get_parameter_counts(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params) for the current model state."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def set_inference_mode(self) -> None:
        """Switch the model to inference mode (disables dropout, enables caching)."""
        FastLanguageModel.for_inference(self.model)

    def save_adapter(self, save_path: str) -> None:
        """Persist only the LoRA adapter weights (not the full model)."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"LoRA adapter saved to: {save_path}")
