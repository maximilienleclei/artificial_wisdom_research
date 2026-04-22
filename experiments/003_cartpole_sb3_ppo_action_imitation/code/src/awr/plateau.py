from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class SmoothedPlateauTracker:
    window_size: int
    best_smoothed_value: float = float("-inf")
    stagnant_updates: int = 0
    _values: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        self._values = deque(maxlen=self.window_size)

    def update(self, value: float) -> float:
        self._values.append(value)
        smoothed_value = sum(self._values) / len(self._values)

        if len(self._values) < self.window_size:
            self.best_smoothed_value = max(self.best_smoothed_value, smoothed_value)
            self.stagnant_updates = 0
            return smoothed_value

        if smoothed_value > self.best_smoothed_value:
            self.best_smoothed_value = smoothed_value
            self.stagnant_updates = 0
        else:
            self.stagnant_updates += 1

        return smoothed_value
