from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import TrainingConfig


@dataclass(slots=True)
class RhythmicSharingModel:
    config: TrainingConfig
    node_adjacency: sparse.csr_matrix
    node_adjacency_dense: np.ndarray
    phase_adjacency: sparse.csr_matrix
    input_weights: np.ndarray
    incidence_transpose: np.ndarray
    incidence_normalization: np.ndarray
    phase_normalization: np.ndarray
    omega_vector: np.ndarray
    nonzero_edge_indices: np.ndarray
    wout: np.ndarray
    training_node_states: np.ndarray
    training_link_phases: np.ndarray
    training_order_parameter: np.ndarray
    training_mean_phase: np.ndarray

    @property
    def input_size(self) -> int:
        return int(self.wout.shape[0])

    @property
    def num_links(self) -> int:
        return int(self.omega_vector.shape[0])


@dataclass(slots=True)
class PredictionResult:
    prediction: np.ndarray
    node_states: np.ndarray
    link_phases: np.ndarray
    order_parameter: np.ndarray
    mean_phase: np.ndarray
    freeze_step: int | None
