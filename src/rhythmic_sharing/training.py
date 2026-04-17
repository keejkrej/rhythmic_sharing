from __future__ import annotations

import numpy as np
from scipy.linalg import pinv

from ._core import (
    advance_nodes,
    advance_training_phases,
    build_topology,
    compute_order_parameters,
    initialize_phases,
    strip_warmup_segments,
    validate_markers,
)
from .config import TrainingConfig
from .data import coerce_time_series
from .model import RhythmicSharingModel


def train(
    train_input: np.ndarray,
    config: TrainingConfig | None = None,
) -> RhythmicSharingModel:
    config = config or TrainingConfig()
    train_input = coerce_time_series(train_input)
    validate_markers(
        config.marker,
        warmup_period=config.warmup_period,
        total_steps=train_input.shape[1],
    )

    train_targets = strip_warmup_segments(
        train_input,
        marker=config.marker,
        warmup_period=config.warmup_period,
    )
    topology = build_topology(train_input.shape[0], config)

    node_state = np.zeros(config.num_nodes_n)
    phases = initialize_phases(topology.num_links)

    effective_steps = train_targets.shape[1]
    states_n = np.zeros((config.num_nodes_n, effective_steps))
    states_phi = np.zeros((topology.num_links, train_input.shape[1]))
    states_phi[:, 0] = phases

    for segment_index, segment_start in enumerate(config.marker):
        for warmup_offset in range(config.warmup_period):
            raw_step = segment_start + warmup_offset
            node_state = advance_nodes(
                node_state,
                phases,
                train_input[:, raw_step],
                topology=topology,
                config=config,
            )
            phases = advance_training_phases(
                node_state,
                phases,
                topology=topology,
                config=config,
            )
            states_phi[:, raw_step + 1] = phases

        output_start = segment_start - (segment_index * config.warmup_period)
        states_n[:, output_start] = node_state

        raw_end = (
            config.marker[segment_index + 1]
            if segment_index + 1 < len(config.marker)
            else train_input.shape[1]
        )
        output_end = output_start + (raw_end - (segment_start + config.warmup_period))

        for output_step in range(output_start, output_end - 1):
            raw_step = output_step + ((segment_index + 1) * config.warmup_period)
            states_n[:, output_step + 1] = advance_nodes(
                states_n[:, output_step],
                phases,
                train_input[:, raw_step],
                topology=topology,
                config=config,
            )
            phases = advance_training_phases(
                states_n[:, output_step + 1],
                phases,
                topology=topology,
                config=config,
            )
            states_phi[:, raw_step + 1] = phases

        node_state = states_n[:, output_end - 1]

    regularized_gram = states_n @ states_n.T
    regularized_gram = regularized_gram + (config.regularization * np.identity(config.num_nodes_n))
    wout = (train_targets @ states_n.T) @ pinv(regularized_gram)

    training_order_parameter, training_mean_phase = compute_order_parameters(states_phi)
    return RhythmicSharingModel(
        config=config,
        node_adjacency=topology.node_adjacency,
        node_adjacency_dense=topology.node_adjacency_dense,
        phase_adjacency=topology.phase_adjacency,
        input_weights=topology.input_weights,
        incidence_transpose=topology.incidence_transpose,
        incidence_normalization=topology.incidence_normalization,
        phase_normalization=topology.phase_normalization,
        omega_vector=topology.omega_vector,
        nonzero_edge_indices=topology.nonzero_edge_indices,
        wout=wout,
        training_node_states=states_n,
        training_link_phases=states_phi,
        training_order_parameter=training_order_parameter,
        training_mean_phase=training_mean_phase,
    )
