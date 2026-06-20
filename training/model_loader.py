from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import LoRAConfig, ModelConfig


class ModelLoader:
    """Responsible for loading the base model and applying LoRA adapters."""

    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig) -> None:
        self.model_config = model_config
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None

    def load_base_model(self) -> Tuple:
        """Load the base model with optional 4-bit quantisation and its tokenizer."""
        quantization_config = None
        if self.model_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if not self.model_config.load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )

        total = sum(p.numel() for p in self.model.parameters())
        print(f"Base model loaded: {self.model_config.name}")
        print(f"  Total parameters: {total:,}")
        return self.model, self.tokenizer

    def apply_lora(self, seed: int = 42) -> None:
        """Wrap the base model with PEFT LoRA adapters (in-place)."""
        if self.model_config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            )

        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)

        total, trainable = self.get_parameter_counts()
        print("LoRA adapters applied.")
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.4f}%)")

    def get_parameter_counts(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params) for the current model state."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def set_inference_mode(self) -> None:
        """Switch the model to inference mode (disables dropout, enables caching)."""
        self.model.eval()
        self.model.config.use_cache = True

    def save_adapter(self, save_path: str) -> None:
        """Persist only the LoRA adapter weights (not the full model)."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"LoRA adapter saved to: {save_path}")
