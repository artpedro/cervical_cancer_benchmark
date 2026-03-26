from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelETATracker:
    dataset: str
    model: str
    total_folds: int
    epochs_per_fold: int

    window_size: int = 10
    train_recent_seconds: list[float] = field(default_factory=list)
    val_recent_seconds: list[float] = field(default_factory=list)
    train_steps_done: int = 0
    val_steps_done: int = 0

    def update_train(self, seconds: float) -> None:
        self.train_recent_seconds.append(float(seconds))
        if len(self.train_recent_seconds) > self.window_size:
            self.train_recent_seconds.pop(0)
        self.train_steps_done += 1

    def update_val(self, seconds: float) -> None:
        self.val_recent_seconds.append(float(seconds))
        if len(self.val_recent_seconds) > self.window_size:
            self.val_recent_seconds.pop(0)
        self.val_steps_done += 1

    @property
    def avg_train_seconds(self) -> float:
        if not self.train_recent_seconds:
            return 0.0
        return sum(self.train_recent_seconds) / len(self.train_recent_seconds)

    @property
    def avg_val_seconds(self) -> float:
        if not self.val_recent_seconds:
            return 0.0
        return sum(self.val_recent_seconds) / len(self.val_recent_seconds)

    @property
    def total_train_steps(self) -> int:
        return self.total_folds * self.epochs_per_fold

    @property
    def total_val_steps(self) -> int:
        return self.total_folds * self.epochs_per_fold

    @property
    def remaining_train_steps(self) -> int:
        return max(0, self.total_train_steps - self.train_steps_done)

    @property
    def remaining_val_steps(self) -> int:
        return max(0, self.total_val_steps - self.val_steps_done)

    @property
    def eta_seconds(self) -> float:
        return (
            self.remaining_train_steps * self.avg_train_seconds
            + self.remaining_val_steps * self.avg_val_seconds
        )

    def eta_hms(self) -> str:
        total = int(round(self.eta_seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

