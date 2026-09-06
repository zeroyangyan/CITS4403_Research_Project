from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
import random

import networkx as nx
import numpy as np


UNAWARE = "Unaware"
BELIEVER = "Believer"
CORRECTED = "Corrected"
RESISTANT = "Resistant"


@dataclass
class ModelConfig:
    n: int = 500
    mean_degree: int = 8
    network_type: str = "small_world"   # random | small_world | scale_free
    ws_rewire_p: float = 0.10

    misinformation_prob: float = 0.15
    correction_prob: float = 0.35
    resistant_fraction: float = 0.05

    intervention: str = "none"          # none | random | centrality
    fact_checker_fraction: float = 0.05
    initial_believers: int = 5

    corrected_share: bool = True
    max_steps: int = 150
    seed: int = 1


def _validate_config(cfg: ModelConfig) -> None:
    if cfg.n < 10:
        raise ValueError("n must be at least 10")
    if cfg.mean_degree < 2 or cfg.mean_degree >= cfg.n:
        raise ValueError("mean_degree must be between 2 and n-1")
    if not 0 <= cfg.misinformation_prob <= 1:
        raise ValueError("misinformation_prob must be in [0, 1]")
    if not 0 <= cfg.correction_prob <= 1:
        raise ValueError("correction_prob must be in [0, 1]")
    if not 0 <= cfg.resistant_fraction < 1:
        raise ValueError("resistant_fraction must be in [0, 1)")
    if not 0 <= cfg.fact_checker_fraction < 1:
        raise ValueError("fact_checker_fraction must be in [0, 1)")
    if cfg.initial_believers < 1:
        raise ValueError("initial_believers must be >= 1")


def generate_network(cfg: ModelConfig) -> nx.Graph:
    """Generate a connected network with approximately comparable mean degree."""
    _validate_config(cfg)
    rng_seed = cfg.seed

    if cfg.network_type == "random":
        p = cfg.mean_degree / (cfg.n - 1)
        # Try a few seeds to obtain a connected graph; otherwise keep largest component
        G = None
        for offset in range(20):
            candidate = nx.erdos_renyi_graph(cfg.n, p, seed=rng_seed + offset)
            if nx.is_connected(candidate):
                G = candidate
                break
        if G is None:
            candidate = nx.erdos_renyi_graph(cfg.n, p, seed=rng_seed)
            largest = max(nx.connected_components(candidate), key=len)
            G = candidate.subgraph(largest).copy()
            # Relabel to contiguous integers
            G = nx.convert_node_labels_to_integers(G)

    elif cfg.network_type == "small_world":
        k = cfg.mean_degree if cfg.mean_degree % 2 == 0 else cfg.mean_degree + 1
        k = min(k, cfg.n - 1 - ((cfg.n - 1) % 2))
        G = nx.watts_strogatz_graph(cfg.n, k, cfg.ws_rewire_p, seed=rng_seed)

    elif cfg.network_type == "scale_free":
        m = max(1, round(cfg.mean_degree / 2))
        m = min(m, cfg.n - 1)
        G = nx.barabasi_albert_graph(cfg.n, m, seed=rng_seed)

    else:
        raise ValueError(f"Unknown network_type: {cfg.network_type}")

    return G


def select_fact_checkers(
    G: nx.Graph,
    strategy: str,
    fraction: float,
    rng: random.Random,
) -> Set[int]:
    """Choose a fixed set of fact-checking agents."""
    if strategy == "none" or fraction <= 0:
        return set()

    count = max(1, round(len(G) * fraction))
    nodes = list(G.nodes())

    if strategy == "random":
        return set(rng.sample(nodes, min(count, len(nodes))))

    if strategy == "centrality":
        ranked = sorted(G.degree, key=lambda x: (-x[1], x[0]))
        return {node for node, _ in ranked[:count]}

    raise ValueError(f"Unknown intervention strategy: {strategy}")


def _prob_at_least_one_success(single_prob: float, exposures: int) -> float:
    if exposures <= 0:
        return 0.0
    return 1.0 - (1.0 - single_prob) ** exposures


def initialise_states(
    G: nx.Graph,
    cfg: ModelConfig,
    fact_checkers: Set[int],
    rng: random.Random,
) -> Tuple[Dict[int, str], Set[int]]:
    """
    Initialise users.
    Fact-checkers start Corrected and cannot become believers.
    Resistant users do not adopt misinformation.
    """
    states = {node: UNAWARE for node in G.nodes()}
    for node in fact_checkers:
        states[node] = CORRECTED

    candidates = [n for n in G.nodes() if n not in fact_checkers]
    resistant_count = round(len(G) * cfg.resistant_fraction)
    resistant_nodes = set(rng.sample(candidates, min(resistant_count, len(candidates))))
    for node in resistant_nodes:
        states[node] = RESISTANT

    seed_candidates = [n for n in candidates if states[n] == UNAWARE]
    if len(seed_candidates) < cfg.initial_believers:
        raise ValueError("Not enough eligible nodes to place initial believers.")

    initial = set(rng.sample(seed_candidates, cfg.initial_believers))
    for node in initial:
        states[node] = BELIEVER

    return states, initial


def run_simulation(cfg: ModelConfig) -> Tuple[List[dict], dict, nx.Graph]:
    """
    Run one synchronous simulation.

    Order within each step:
    1) Current believers expose unaware neighbours to misinformation.
    2) Fact-checkers and (optionally) corrected users expose believers to corrections.
    3) All resulting state changes are applied simultaneously.
    """
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    G = generate_network(cfg)
    fact_checkers = select_fact_checkers(
        G, cfg.intervention, cfg.fact_checker_fraction, rng
    )
    states, initial_believers = initialise_states(G, cfg, fact_checkers, rng)

    ever_believed = set(initial_believers)
    history: List[dict] = []

    def record(step: int) -> None:
        counts = {
            UNAWARE: 0,
            BELIEVER: 0,
            CORRECTED: 0,
            RESISTANT: 0,
        }
        for state in states.values():
            counts[state] += 1
        history.append(
            {
                "step": step,
                "unaware": counts[UNAWARE],
                "believer": counts[BELIEVER],
                "corrected": counts[CORRECTED],
                "resistant": counts[RESISTANT],
                "ever_believed": len(ever_believed),
            }
        )

    record(0)

    for step in range(1, cfg.max_steps + 1):
        believers = {n for n, s in states.items() if s == BELIEVER}
        if not believers:
            break

        misinformation_exposures: Dict[int, int] = {}
        correction_exposures: Dict[int, int] = {}

        # Misinformation exposures
        for source in believers:
            for nbr in G.neighbors(source):
                if states[nbr] == UNAWARE and nbr not in fact_checkers:
                    misinformation_exposures[nbr] = misinformation_exposures.get(nbr, 0) + 1

        # Correction exposures
        correction_sources = set(fact_checkers)
        if cfg.corrected_share:
            correction_sources.update(
                n for n, s in states.items() if s == CORRECTED
            )

        for source in correction_sources:
            for nbr in G.neighbors(source):
                if states[nbr] == BELIEVER:
                    correction_exposures[nbr] = correction_exposures.get(nbr, 0) + 1

        to_believer: Set[int] = set()
        to_corrected: Set[int] = set()

        for node, exposures in misinformation_exposures.items():
            p = _prob_at_least_one_success(cfg.misinformation_prob, exposures)
            if rng.random() < p:
                to_believer.add(node)

        for node, exposures in correction_exposures.items():
            p = _prob_at_least_one_success(cfg.correction_prob, exposures)
            if rng.random() < p:
                to_corrected.add(node)

        # Simultaneous update
        for node in to_believer:
            if states[node] == UNAWARE:
                states[node] = BELIEVER
                ever_believed.add(node)

        for node in to_corrected:
            if states[node] == BELIEVER:
                states[node] = CORRECTED

        record(step)

    believer_series = [row["believer"] for row in history]
    peak_believers = max(believer_series)
    time_to_peak = believer_series.index(peak_believers)
    extinction_time: Optional[int] = None
    if believer_series[-1] == 0:
        extinction_time = history[-1]["step"]

    metrics = {
        **asdict(cfg),
        "actual_n": len(G),
        "actual_mean_degree": (
            sum(dict(G.degree()).values()) / len(G) if len(G) else 0.0
        ),
        "num_fact_checkers": len(fact_checkers),
        "final_reach": len(ever_believed) / len(G),
        "peak_believers": peak_believers,
        "peak_believer_fraction": peak_believers / len(G),
        "time_to_peak": time_to_peak,
        "extinction_time": extinction_time,
        "ended_with_active_believers": believer_series[-1] > 0,
    }

    return history, metrics, G
