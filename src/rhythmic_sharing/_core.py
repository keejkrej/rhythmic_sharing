from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from .config import InferenceConfig, TrainingConfig
from .model import RhythmicSharingModel


@dataclass(slots=True)
class PreparedTopology:
    node_adjacency: sparse.csr_matrix
    node_adjacency_dense: np.ndarray
    phase_adjacency: sparse.csr_matrix
    input_weights: np.ndarray
    incidence_transpose: np.ndarray
    incidence_normalization: np.ndarray
    phase_normalization: np.ndarray
    omega_vector: np.ndarray
    nonzero_edge_indices: np.ndarray

    @property
    def num_links(self) -> int:
        return int(self.omega_vector.shape[0])


def incidence_matrix(
    graph: nx.Graph,
    nodelist: list[int] | None = None,
    edgelist: list[tuple[int, int]] | None = None,
    *,
    oriented: bool = False,
    weight: str | None = None,
) -> sparse.csc_array:
    if nodelist is None:
        nodelist = list(graph)
    if edgelist is None:
        if graph.is_multigraph():
            edgelist = list(graph.edges(keys=True))
        else:
            edgelist = list(graph.edges())

    matrix = sparse.lil_array((len(nodelist), len(edgelist)))
    node_index = {node: i for i, node in enumerate(nodelist)}

    for edge_index, edge in enumerate(edgelist):
        u, v = edge[:2]

        if u == v:
            matrix[node_index[u], edge_index] = 1
            continue

        ui = node_index[u]
        vi = node_index[v]
        edge_weight = 1 if weight is None else graph[u][v].get(weight, 1)

        if oriented:
            matrix[ui, edge_index] = -edge_weight
            matrix[vi, edge_index] = edge_weight
        else:
            matrix[ui, edge_index] = edge_weight
            matrix[vi, edge_index] = edge_weight

    return matrix.asformat("csc")


def compute_order_parameters(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rx = np.average(np.cos(phases), axis=0)
    ry = np.average(np.sin(phases), axis=0)
    order_parameter = np.sqrt((rx**2) + (ry**2))
    mean_phase = np.arctan2(ry, rx)
    return order_parameter, mean_phase


def initialize_phases(num_links: int, *, seed_offset: int = 0) -> np.ndarray:
    phases = np.zeros(num_links)
    for index in range(num_links):
        rng = np.random.RandomState(index + seed_offset)
        phases[index] = float(rng.rand(1)[0] * 2 * np.pi)
    return phases


def build_topology(input_size: int, config: TrainingConfig) -> PreparedTopology:
    node_density = min(1.0, config.average_degree_n / config.num_nodes_n)
    node_adjacency = sparse.random(
        config.num_nodes_n,
        config.num_nodes_n,
        density=node_density,
        random_state=config.data_seed,
        format="csr",
    )
    node_adjacency = (2 * node_adjacency) - node_adjacency.ceil()
    largest_eigenvalue = linalg.eigs(
        node_adjacency,
        k=1,
        return_eigenvectors=False,
    )[0]
    node_adjacency = (
        config.spectral_radius_n / np.abs(largest_eigenvalue)
    ) * node_adjacency
    node_adjacency = node_adjacency.tocsr()
    node_adjacency_dense = node_adjacency.toarray()

    nonzero_edge_indices = np.flatnonzero(node_adjacency_dense.ravel())
    edge_coordinates = list(zip(*np.nonzero(node_adjacency_dense)))

    graph = nx.DiGraph()
    graph.add_nodes_from(range(config.num_nodes_n))
    graph.add_edges_from(edge_coordinates)

    incidence = incidence_matrix(
        graph,
        nodelist=list(range(config.num_nodes_n)),
        edgelist=edge_coordinates,
        oriented=False,
    ).toarray()
    incidence_transpose = incidence.T
    incidence_normalization = np.count_nonzero(incidence_transpose, axis=1).astype(
        float
    )
    incidence_normalization[incidence_normalization == 0.0] = 1.0

    qq = int(np.floor(config.num_nodes_n / input_size))
    input_weights = np.zeros((config.num_nodes_n, input_size))
    for input_index in range(input_size):
        rng = np.random.RandomState(input_index)
        values = (-1 + (2 * rng.rand(qq))) * config.input_weight_n
        start = input_index * qq
        stop = (input_index + 1) * qq
        input_weights[start:stop, input_index] = values

    num_links = len(edge_coordinates)
    phase_density = 0.0
    if num_links:
        phase_density = min(1.0, config.average_degree_links / num_links)
    phase_adjacency = sparse.random(
        num_links,
        num_links,
        density=phase_density,
        random_state=config.data_seed + 3,
        format="csr",
    )
    phase_adjacency = phase_adjacency.ceil().tocsr()
    phase_normalization = np.asarray(phase_adjacency.sum(axis=1)).ravel()
    phase_normalization[phase_normalization == 0.0] = 1000.0

    omega_mask = sparse.random(
        1,
        num_links,
        density=config.modpop,
        random_state=config.data_seed + 5,
        format="csr",
    )
    omega_mask = np.ceil(omega_mask.toarray()).ravel()

    if config.discrete_omega0:
        omega_vector = omega_mask * config.omega0
    else:
        if config.omega0_mean is None or config.omega0_spread is None:
            raise ValueError(
                "omega0_mean and omega0_spread are required when discrete_omega0=False."
            )
        omega_vector = omega_mask.copy()
        active_indices = np.flatnonzero(omega_vector)
        rng = np.random.default_rng(config.data_seed + 5)
        omega_vector[active_indices] = rng.normal(
            loc=config.omega0_mean,
            scale=config.omega0_spread,
            size=active_indices.shape[0],
        )

    return PreparedTopology(
        node_adjacency=node_adjacency,
        node_adjacency_dense=node_adjacency_dense,
        phase_adjacency=phase_adjacency,
        input_weights=input_weights,
        incidence_transpose=incidence_transpose,
        incidence_normalization=incidence_normalization,
        phase_normalization=phase_normalization,
        omega_vector=omega_vector,
        nonzero_edge_indices=nonzero_edge_indices,
    )


def phase_vector_to_matrix(
    phases: np.ndarray,
    nonzero_edge_indices: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    phase_matrix = np.zeros(num_nodes * num_nodes)
    phase_matrix[nonzero_edge_indices] = phases
    return phase_matrix.reshape(num_nodes, num_nodes)


def advance_nodes(
    node_state: np.ndarray,
    phases: np.ndarray,
    input_vector: np.ndarray,
    *,
    topology: PreparedTopology | RhythmicSharingModel,
    config: TrainingConfig,
) -> np.ndarray:
    phase_matrix = phase_vector_to_matrix(
        phases,
        topology.nonzero_edge_indices,
        config.num_nodes_n,
    )
    modulation = 1 - (config.modstrength / 2.0) * (1 + np.sin(phase_matrix))
    activation = (topology.node_adjacency_dense * modulation).dot(node_state)
    activation = activation + (topology.input_weights @ input_vector) + config.bias_n
    return (config.leakage_n * node_state) + (
        (1 - config.leakage_n) * np.tanh(activation)
    )


def phase_forcing(
    node_state: np.ndarray,
    phases: np.ndarray,
    *,
    topology: PreparedTopology | RhythmicSharingModel,
    config: TrainingConfig,
) -> np.ndarray:
    rx = topology.phase_adjacency.dot(np.cos(phases)) / topology.phase_normalization
    ry = topology.phase_adjacency.dot(np.sin(phases)) / topology.phase_normalization
    local_mean_phase = np.arctan2(ry, rx)

    node_drive = topology.incidence_transpose @ ((node_state + 1.0) / 2.0)
    node_drive = node_drive / topology.incidence_normalization

    return np.sin((local_mean_phase - phases) + config.bias_phase) * (
        config.epsilon_1 + (config.epsilon_2 * node_drive)
    )


def advance_training_phases(
    node_state: np.ndarray,
    phases: np.ndarray,
    *,
    topology: PreparedTopology,
    config: TrainingConfig,
) -> np.ndarray:
    return phases + config.tau * (
        topology.omega_vector + phase_forcing(node_state, phases, topology=topology, config=config)
    )


def advance_prediction_phases(
    *,
    step: int,
    node_state: np.ndarray,
    phases: np.ndarray,
    error_current: float,
    phase_frozen: bool,
    model: RhythmicSharingModel,
    inference_config: InferenceConfig,
) -> tuple[np.ndarray, bool]:
    training_config = model.config
    if step < inference_config.phase_override_step:
        phase_next = phases + training_config.tau * (
            model.omega_vector
            + phase_forcing(
                node_state,
                phases,
                topology=model,
                config=training_config,
            )
        )
        return phase_next, phase_frozen

    if phase_frozen:
        return phases.copy(), phase_frozen

    _, global_mean_phase = compute_order_parameters(phases)
    if (
        abs(global_mean_phase - inference_config.freeze_mean_phase)
        < inference_config.freeze_mean_phase_tolerance
        and error_current < inference_config.freeze_error_tolerance
    ):
        return phases.copy(), True

    return phases + (training_config.tau * training_config.omega0), phase_frozen


def strip_warmup_segments(
    train_input: np.ndarray,
    *,
    marker: tuple[int, ...],
    warmup_period: int,
) -> np.ndarray:
    markers = np.asarray(marker, dtype=int)
    output_steps = train_input.shape[1] - (markers.shape[0] * warmup_period)
    if output_steps <= 0:
        raise ValueError("Warmup removes every training timestep.")

    output = np.zeros((train_input.shape[0], output_steps))
    for segment_index, segment_start in enumerate(markers):
        raw_start = segment_start + warmup_period
        raw_end = (
            markers[segment_index + 1]
            if segment_index + 1 < markers.shape[0]
            else train_input.shape[1]
        )
        output_start = segment_start - (segment_index * warmup_period)
        output_end = output_start + (raw_end - raw_start)
        output[:, output_start:output_end] = train_input[:, raw_start:raw_end]

    return output


def validate_markers(
    markers: tuple[int, ...],
    *,
    warmup_period: int,
    total_steps: int,
) -> None:
    previous = None
    for marker in markers:
        if marker < 0 or marker >= total_steps:
            raise ValueError("marker entries must fall inside the training series.")
        if previous is not None and marker <= previous:
            raise ValueError("marker entries must be strictly increasing.")
        previous = marker

    for segment_index, marker in enumerate(markers):
        raw_end = markers[segment_index + 1] if segment_index + 1 < len(markers) else total_steps
        if raw_end - marker <= warmup_period:
            raise ValueError(
                "Each marker-delimited segment must contain more steps than warmup_period."
            )
