import re
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from training import DataConfig


class DatasetLoader:
    """Handles loading, splitting, and formatting of the SQL dataset."""

    def __init__(self, config: DataConfig, system_prompt: str) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self.train_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def load(self) -> None:
        """Download, shuffle and split the dataset into train/test partitions."""
        dataset = load_dataset(self.config.dataset_name, split="train")
        dataset = dataset.shuffle(seed=self.config.seed)
        self.train_data = dataset.select(range(self.config.num_train))
        self.test_data = dataset.select(
            range(self.config.num_train, self.config.num_train + self.config.num_test)
        )
        print(f"Dataset loaded: {self.config.dataset_name}")
        print(f"  Train: {len(self.train_data):,} | Test: {len(self.test_data):,}")

    def build_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Construct the chat message list for a single example (no answer)."""
        user_msg = f"Schema:\n{example['context']}\n\nQuestion: {example['question']}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def build_training_text(self, example: Dict[str, Any], tokenizer) -> str:
        """Build the full conversation string including the expected answer for SFT."""
        messages = self.build_messages(example)
        messages.append({"role": "assistant", "content": example["answer"]})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def prepare_training_dataset(self, tokenizer) -> Dataset:
        """Map raw train data into tokenizer-formatted text strings."""
        return self.train_data.map(
            lambda ex: {"text": self.build_training_text(ex, tokenizer)},
            remove_columns=self.train_data.column_names,
        )

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return a summary dict of dataset sizes and source."""
        return {
            "dataset_name": self.config.dataset_name,
            "train_size": len(self.train_data) if self.train_data else 0,
            "test_size": len(self.test_data) if self.test_data else 0,
            "seed": self.config.seed,
        }

    def get_sample_examples(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the first n training examples as plain dicts."""
        return [
            {
                "question": self.train_data[i]["question"],
                "context": self.train_data[i]["context"],
                "answer": self.train_data[i]["answer"],
            }
            for i in range(min(n, len(self.train_data)))
        ]

    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Lowercase, strip markdown fences and collapse whitespace for fair comparison."""
        sql = sql.strip().lower()
        sql = sql.rstrip(";")
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*$", "", sql)
        sql = re.sub(r"\s+", " ", sql)
        return sql.strip()
