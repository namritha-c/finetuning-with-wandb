from typing import Any, Dict, List
import time

import torch

from training import DatasetLoader


class ModelEvaluator:
    """Runs exact-match SQL evaluation against a test set."""

    def __init__(self, model, tokenizer, data_loader: DatasetLoader) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader

    def generate_sql(self, example: Dict[str, Any], max_new_tokens: int = 256) -> str:
        """Run a single inference pass and return the raw generated SQL string."""
        messages = self.data_loader.build_messages(example)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def evaluate(self, test_data, label: str = "Model") -> Dict[str, Any]:
        """
        Evaluate the model over the entire test set using exact-match accuracy.

        Returns a dict with accuracy, correct count, timing, and per-example results.
        """
        self.model.eval()
        self.model.config.use_cache = True
        correct = 0
        results: List[Dict[str, Any]] = []
        start = time.time()

        for i, example in enumerate(test_data):
            predicted = self.generate_sql(example)
            expected = example["answer"]

            pred_norm = DatasetLoader.normalize_sql(predicted)
            exp_norm = DatasetLoader.normalize_sql(expected)
            match = pred_norm == exp_norm
            if match:
                correct += 1

            results.append({
                "question": example["question"],
                "schema": example["context"],
                "expected": expected,
                "predicted": predicted,
                "match": match,
            })

            if (i + 1) % 50 == 0:
                acc_so_far = correct / (i + 1) * 100
                print(f"  [{label}] {i + 1}/{len(test_data)} — accuracy: {acc_so_far:.1f}%")

        elapsed = time.time() - start
        accuracy = correct / len(test_data) * 100
        print(f"\n{'=' * 50}")
        print(f"{label} — Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
        print(f"Time: {elapsed:.0f}s  ({elapsed / len(test_data):.1f}s per example)")
        print(f"{'=' * 50}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_data),
            "elapsed_seconds": elapsed,
            "results": results,
        }

    def print_sample_outputs(
        self, eval_results: Dict[str, Any], n: int = 5, label: str = "Model"
    ) -> None:
        """Print the first n predictions with CORRECT / WRONG labels."""
        print(f"\nSample outputs from {label}:\n")
        for i, r in enumerate(eval_results["results"][:n]):
            status = "CORRECT" if r["match"] else "WRONG"
            print(f"--- Example {i + 1} [{status}] ---")
            print(f"  Question: {r['question']}")
            print(f"  Expected: {r['expected']}")
            print(f"  Got:      {r['predicted'][:200]}")
            print()
