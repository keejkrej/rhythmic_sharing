from __future__ import annotations

import numpy as np

from ._core import (
    advance_nodes,
    advance_prediction_phases,
    compute_order_parameters,
    initialize_phases,
)
from .config import InferenceConfig
from .data import coerce_time_series
from .model import PredictionResult, RhythmicSharingModel


def predict(
    model: RhythmicSharingModel,
    test_data: np.ndarray,
    config: InferenceConfig | None = None,
) -> PredictionResult:
    config = config or InferenceConfig()
    test_data = coerce_time_series(test_data)

    pred_warmup_period = (
        model.config.warmup_period
        if config.pred_warmup_period is None
        else config.pred_warmup_period
    )
    if pred_warmup_period >= test_data.shape[1]:
        raise ValueError("pred_warmup_period must be smaller than the number of test steps.")
    if test_data.shape[0] != model.input_size:
        raise ValueError("test_data dimensionality must match the trained readout output size.")

    prediction = np.zeros((model.input_size, test_data.shape[1]))
    node_state = np.zeros(model.config.num_nodes_n)
    states_n = np.zeros((model.config.num_nodes_n, test_data.shape[1]))

    phases = initialize_phases(model.num_links, seed_offset=config.phase_init_seed_offset)
    states_phi = np.zeros((model.num_links, test_data.shape[1]))
    states_phi[:, 0] = phases

    prediction[:, 0] = model.wout @ node_state

    phase_frozen = False
    freeze_step = None

    for step in range(pred_warmup_period):
        error_current = float(np.sum(prediction[:, step] - test_data[:, step]) ** 2)
        node_state = advance_nodes(
            node_state,
            phases,
            test_data[:, step],
            topology=model,
            config=model.config,
        )
        phases, phase_frozen = advance_prediction_phases(
            step=step,
            node_state=node_state,
            phases=phases,
            error_current=error_current,
            phase_frozen=phase_frozen,
            model=model,
            inference_config=config,
        )
        if phase_frozen and freeze_step is None:
            freeze_step = step

        states_phi[:, step + 1] = phases
        states_n[:, step + 1] = node_state
        prediction[:, step + 1] = model.wout @ node_state

    for step in range(pred_warmup_period, test_data.shape[1] - 1):
        error_current = float(np.sum(prediction[:, step] - test_data[:, step]) ** 2)
        node_state = advance_nodes(
            node_state,
            phases,
            prediction[:, step],
            topology=model,
            config=model.config,
        )
        phases, phase_frozen = advance_prediction_phases(
            step=step,
            node_state=node_state,
            phases=phases,
            error_current=error_current,
            phase_frozen=phase_frozen,
            model=model,
            inference_config=config,
        )
        if phase_frozen and freeze_step is None:
            freeze_step = step

        states_phi[:, step + 1] = phases
        states_n[:, step + 1] = node_state
        prediction[:, step + 1] = model.wout @ node_state

    order_parameter, mean_phase = compute_order_parameters(states_phi)
    return PredictionResult(
        prediction=prediction,
        node_states=states_n,
        link_phases=states_phi,
        order_parameter=order_parameter,
        mean_phase=mean_phase,
        freeze_step=freeze_step,
    )
