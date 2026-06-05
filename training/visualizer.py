from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


class Visualizer:
    """Generates and saves all training and evaluation plots."""

    @staticmethod
    def plot_training_loss(
        log_history: List[Dict[str, Any]],
        save_path: str = "training_loss.png",
    ) -> Tuple[List[int], List[float]]:
        """Plot training loss over steps and persist to disk."""
        steps = [e["step"] for e in log_history if "loss" in e]
        losses = [e["loss"] for e in log_history if "loss" in e]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, linewidth=2, color="#2196F3")
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("LoRA Fine-Tuning Loss Curve", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if losses:
            print(f"Loss: {losses[0]:.3f} → {losses[-1]:.3f}  (saved: {save_path})")
        return steps, losses

    @staticmethod
    def plot_accuracy_comparison(
        base_acc: float,
        ft_acc: float,
        save_path: str = "accuracy_comparison.png",
    ) -> None:
        """Side-by-side bar chart comparing base and fine-tuned accuracy."""
        improvement = ft_acc - base_acc
        labels = ["Base Model\n(Before Fine-Tuning)", "Fine-Tuned Model\n(After LoRA)"]
        colors = ["#EF5350", "#66BB6A"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, [base_acc, ft_acc], color=colors, width=0.5,
                      edgecolor="white", linewidth=2)
        for bar, acc in zip(bars, [base_acc, ft_acc]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=16, fontweight="bold",
            )
        ax.set_ylabel("Exact-Match Accuracy (%)", fontsize=13)
        ax.set_title("Text-to-SQL: Before vs After LoRA Fine-Tuning",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_ylim(0, max(base_acc, ft_acc) + 15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.annotate(
            f"+{improvement:.1f}%",
            xy=(1, ft_acc), xytext=(1.35, (base_acc + ft_acc) / 2),
            fontsize=14, fontweight="bold", color="#1565C0",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2),
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Accuracy comparison saved: {save_path}")

    @staticmethod
    def plot_summary(
        base_acc: float,
        ft_acc: float,
        total_params: int,
        trainable_params: int,
        train_runtime_seconds: float,
        save_path: str = "summary.png",
    ) -> None:
        """Three-panel summary: accuracy bars, parameter pie, and training time."""
        frozen_params = total_params - trainable_params
        train_time_min = train_runtime_seconds / 60

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].bar(["Before", "After"], [base_acc, ft_acc],
                    color=["#EF5350", "#66BB6A"], width=0.5)
        axes[0].set_title("Accuracy (%)", fontsize=13, fontweight="bold")
        axes[0].set_ylim(0, 100)
        for j, v in enumerate([base_acc, ft_acc]):
            axes[0].text(j, v + 2, f"{v:.1f}%", ha="center",
                         fontweight="bold", fontsize=12)

        axes[1].pie(
            [trainable_params, frozen_params],
            labels=[f"LoRA\n{trainable_params / 1e6:.1f}M",
                    f"Frozen\n{frozen_params / 1e6:.0f}M"],
            colors=["#FF9800", "#E0E0E0"],
            startangle=90,
            textprops={"fontsize": 11},
            explode=(0.1, 0),
        )
        axes[1].set_title(
            f"Parameters Trained ({trainable_params / total_params * 100:.2f}%)",
            fontsize=13, fontweight="bold",
        )

        axes[2].barh(["Training\nTime"], [train_time_min], color="#42A5F5", height=0.4)
        axes[2].set_xlabel("Minutes", fontsize=11)
        axes[2].set_title("Time to Fine-Tune", fontsize=13, fontweight="bold")
        axes[2].text(train_time_min + 0.3, 0, f"{train_time_min:.1f} min",
                     va="center", fontsize=12, fontweight="bold")
        axes[2].set_xlim(0, train_time_min * 1.5)

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("LoRA Fine-Tuning Summary", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Summary chart saved: {save_path}")
