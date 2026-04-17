from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    tau: float = 1.0
    average_degree_n: int = 10
    num_nodes_n: int = 100
    discrete_omega0: bool = True
    omega0: float = 0.01
    omega0_mean: float | None = None
    omega0_spread: float | None = None
    average_degree_links: int | None = None
    input_weight_n: float = 120e-2
    epsilon_1: float = -0.2
    epsilon_2: float = 0.6
    leakage_n: float = 0.0
    spectral_radius_n: float = 0.6
    bias_n: float = 0.0
    bias_phase: float = 0.0
    modpop: float = 1.0
    modstrength: float = 0.6
    regularization: float = 1e-20
    data_seed: int = 0
    marker: tuple[int, ...] = (0,)
    warmup_period: int = 200

    def __post_init__(self) -> None:
        self.marker = tuple(int(value) for value in self.marker)
        if not self.marker:
            raise ValueError("marker must contain at least one segment start.")
        if self.average_degree_links is None:
            self.average_degree_links = int(self.num_nodes_n / 2)


@dataclass(slots=True)
class InferenceConfig:
    pred_warmup_period: int | None = None
    phase_override_step: int = 650
    freeze_mean_phase: float = math.pi
    freeze_mean_phase_tolerance: float = 1e-3
    freeze_error_tolerance: float = 1e-3
    phase_init_seed_offset: int = 2
