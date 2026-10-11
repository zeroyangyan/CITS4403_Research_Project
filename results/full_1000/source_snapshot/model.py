from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import random
import hashlib
import json

import networkx as nx
import numpy as np


UNAWARE = "Unaware"
BELIEVER = "Believer"
CORRECTED = "Corrected"
RESISTANT = "Resistant"


@dataclass
class ModelConfig:
    n: int = 1000
    mean_degree: int = 8
    network_type: str = "small_world"
    ws_rewire_p: float = 0.10

    misinformation_prob: float = 0.15
    correction_prob: float = 0.35
    resistant_fraction: float = 0.05

    intervention: str = "none"
    fact_checker_fraction: float = 0.05

    # 默认使用实际节点数的1%。
    # initial_believer_fraction=None 时才使用固定人数。
    initial_believers: int = 10
    initial_believer_fraction: Optional[float] = 0.01

    corrected_share: bool = True
    max_steps: int = 300
    seed: int = 1
    heterogeneity: str = "homogeneous"


STRENGTHS = {
    "homogeneous": 0.0,
    "low": 0.25,
    "high": 0.90,
}


def _validate_config(cfg: ModelConfig) -> None:
    if cfg.heterogeneity not in STRENGTHS:
        raise ValueError("Unknown heterogeneity level")

    if not 0 <= cfg.ws_rewire_p <= 1:
        raise ValueError("ws_rewire_p must be in [0, 1]")

    if cfg.intervention not in ("none", "random", "centrality"):
        raise ValueError("Unknown intervention")

    if cfg.max_steps < 1 or cfg.seed < 0:
        raise ValueError("max_steps must be positive and seed nonnegative")

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

    if (
        cfg.initial_believer_fraction is not None
        and not 0 < cfg.initial_believer_fraction < 1
    ):
        raise ValueError("initial_believer_fraction must be in (0, 1)")

    if cfg.initial_believers < 1:
        raise ValueError("initial_believers must be >= 1")


def resolve_seeds(cfg: ModelConfig) -> dict:
    strategy_id = {
        "none": 0,
        "random": 1,
        "centrality": 2,
    }[cfg.intervention]

    def derive(channel: int, strategy: int = 0) -> int:
        return int(
            np.random.SeedSequence(
                [cfg.seed, channel, strategy]
            ).generate_state(1)[0]
        )

    # 网络和用户属性不依赖策略。
    # 角色选择、初始身份和状态转移使用策略各自的随机流。
    return {
        "network_seed": derive(0),
        "attribute_seed": derive(1),
        "selection_seed": derive(2, strategy_id),
        "initialisation_seed": derive(3, strategy_id),
        "transition_seed": derive(4, strategy_id),
    }


def make_probabilities(nodes, mean, level, rng):
    width = STRENGTHS[level] * min(mean, 1.0 - mean)

    if width == 0:
        return {node: mean for node in nodes}

    # 成对构造概率，让全部节点的平均值保持不变。
    values = []

    for _ in range(len(nodes) // 2):
        difference = rng.uniform(0.0, width)
        values.extend([
            mean - difference,
            mean + difference,
        ])

    if len(nodes) % 2:
        values.append(mean)

    rng.shuffle(values)
    return dict(zip(nodes, values))


def graph_fingerprint(G: nx.Graph) -> str:
    payload = {
        "nodes": sorted(G.nodes()),
        "edges": sorted(
            (min(a, b), max(a, b))
            for a, b in G.edges()
        ),
    }

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def generate_network(cfg: ModelConfig) -> nx.Graph:
    _validate_config(cfg)
    network_seed = resolve_seeds(cfg)["network_seed"]

    if cfg.network_type == "random":
        edge_probability = cfg.mean_degree / (cfg.n - 1)
        G = None

        # 保留原模型的连通网络生成规则。
        for offset in range(20):
            candidate = nx.erdos_renyi_graph(
                cfg.n,
                edge_probability,
                seed=network_seed + offset,
            )

            if nx.is_connected(candidate):
                G = candidate
                G.graph["realised_network_seed"] = network_seed + offset
                break

        if G is None:
            candidate = nx.erdos_renyi_graph(
                cfg.n,
                edge_probability,
                seed=network_seed,
            )
            largest_component = max(
                nx.connected_components(candidate),
                key=len,
            )
            G = candidate.subgraph(largest_component).copy()
            G = nx.convert_node_labels_to_integers(G)
            G.graph["realised_network_seed"] = network_seed

    elif cfg.network_type == "small_world":
        k = (
            cfg.mean_degree
            if cfg.mean_degree % 2 == 0
            else cfg.mean_degree + 1
        )
        k = min(k, cfg.n - 1 - ((cfg.n - 1) % 2))

        G = nx.watts_strogatz_graph(
            cfg.n,
            k,
            cfg.ws_rewire_p,
            seed=network_seed,
        )

    elif cfg.network_type == "scale_free":
        m = max(1, round(cfg.mean_degree / 2))
        m = min(m, cfg.n - 1)

        G = nx.barabasi_albert_graph(
            cfg.n,
            m,
            seed=network_seed,
        )

    else:
        raise ValueError(f"Unknown network_type: {cfg.network_type}")

    return G


def select_fact_checkers(
    G: nx.Graph,
    strategy: str,
    fraction: float,
    rng: random.Random,
) -> Set[int]:
    if strategy == "none" or fraction <= 0:
        return set()

    count = max(1, round(len(G) * fraction))
    nodes = sorted(G.nodes())

    if strategy == "random":
        return set(rng.sample(nodes, min(count, len(nodes))))

    if strategy == "centrality":
        # 这里的centrality具体指度数定向选择。
        ranked = sorted(
            G.degree,
            key=lambda item: (-item[1], item[0]),
        )
        return {node for node, _ in ranked[:count]}

    raise ValueError(f"Unknown intervention strategy: {strategy}")


def _prob_at_least_one_success(
    single_prob: float,
    exposures: int,
) -> float:
    if exposures <= 0:
        return 0.0

    return 1.0 - (1.0 - single_prob) ** exposures


def initialise_states(
    G: nx.Graph,
    cfg: ModelConfig,
    fact_checkers: Set[int],
    rng: random.Random,
) -> Tuple[Dict[int, str], Set[int]]:
    states = {node: UNAWARE for node in G.nodes()}

    for node in fact_checkers:
        states[node] = CORRECTED

    candidates = [
        node
        for node in sorted(G.nodes())
        if node not in fact_checkers
    ]

    resistant_count = round(len(G) * cfg.resistant_fraction)
    resistant_nodes = set(
        rng.sample(
            candidates,
            min(resistant_count, len(candidates)),
        )
    )

    for node in resistant_nodes:
        states[node] = RESISTANT

    seed_candidates = [
        node
        for node in candidates
        if states[node] == UNAWARE
    ]

    if cfg.initial_believer_fraction is not None:
        initial_count = max(
            1,
            round(len(G) * cfg.initial_believer_fraction),
        )
    else:
        initial_count = cfg.initial_believers

    if len(seed_candidates) < initial_count:
        raise ValueError(
            "Not enough eligible nodes to place initial believers."
        )

    initial = set(rng.sample(seed_candidates, initial_count))

    for node in initial:
        states[node] = BELIEVER

    return states, initial


def run_simulation(
    cfg: ModelConfig,
) -> Tuple[List[dict], dict, nx.Graph]:
    seeds = resolve_seeds(cfg)

    selection_rng = random.Random(seeds["selection_seed"])
    initial_rng = random.Random(seeds["initialisation_seed"])
    transition_rng = random.Random(seeds["transition_seed"])

    G = generate_network(cfg)
    nodes = sorted(G.nodes())

    attribute_rng = random.Random(seeds["attribute_seed"])

    adoption = make_probabilities(
        nodes,
        cfg.misinformation_prob,
        cfg.heterogeneity,
        attribute_rng,
    )
    correction = make_probabilities(
        nodes,
        cfg.correction_prob,
        cfg.heterogeneity,
        attribute_rng,
    )

    fact_checkers = select_fact_checkers(
        G,
        cfg.intervention,
        cfg.fact_checker_fraction,
        selection_rng,
    )

    states, initial_believers = initialise_states(
        G,
        cfg,
        fact_checkers,
        initial_rng,
    )

    resistant_nodes = {
        node
        for node, state in states.items()
        if state == RESISTANT
    }

    ordinary = sorted(
        set(nodes) - fact_checkers - resistant_nodes
    )

    ever_believed = set(initial_believers)
    history = []

    def record(step):
        counts = {
            UNAWARE: 0,
            BELIEVER: 0,
            CORRECTED: 0,
            RESISTANT: 0,
        }

        for state in states.values():
            counts[state] += 1

        history.append({
            "step": step,
            "unaware": counts[UNAWARE],
            "believer": counts[BELIEVER],
            "corrected": counts[CORRECTED],
            "resistant": counts[RESISTANT],
            "ever_believed": len(ever_believed),
        })

    record(0)

    for step in range(1, cfg.max_steps + 1):
        believers = {
            node
            for node, state in states.items()
            if state == BELIEVER
        }

        if not believers:
            break

        misinformation_exposures = {}
        correction_exposures = {}

        for source in sorted(believers):
            for neighbour in sorted(G.neighbors(source)):
                if (
                    states[neighbour] == UNAWARE
                    and neighbour not in fact_checkers
                ):
                    misinformation_exposures[neighbour] = (
                        misinformation_exposures.get(neighbour, 0) + 1
                    )

        correction_sources = set(fact_checkers)

        if cfg.corrected_share:
            correction_sources.update(
                node
                for node, state in states.items()
                if state == CORRECTED
            )

        for source in sorted(correction_sources):
            for neighbour in sorted(G.neighbors(source)):
                if states[neighbour] == BELIEVER:
                    correction_exposures[neighbour] = (
                        correction_exposures.get(neighbour, 0) + 1
                    )

        to_believer = set()
        to_corrected = set()

        for node, exposures in sorted(misinformation_exposures.items()):
            probability = _prob_at_least_one_success(
                adoption[node],
                exposures,
            )

            if transition_rng.random() < probability:
                to_believer.add(node)

        for node, exposures in sorted(correction_exposures.items()):
            probability = _prob_at_least_one_success(
                correction[node],
                exposures,
            )

            if transition_rng.random() < probability:
                to_corrected.add(node)

        # 同步更新，新相信者从下一步才开始传播。
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

    extinction_time = None

    if believer_series[-1] == 0:
        extinction_time = history[-1]["step"]

    metrics = {
        **asdict(cfg),
        **seeds,
        "heterogeneity_strength": STRENGTHS[cfg.heterogeneity],
        "attribute_distribution": "paired_symmetric_uniform",
        "adoption_mean": float(np.mean(list(adoption.values()))),
        "adoption_std": float(np.std(list(adoption.values()))),
        "correction_mean": float(np.mean(list(correction.values()))),
        "correction_std": float(np.std(list(correction.values()))),
        "ordinary_adoption_mean": float(
            np.mean([adoption[node] for node in ordinary])
        ),
        "ordinary_correction_mean": float(
            np.mean([correction[node] for node in ordinary])
        ),
        "initialisation_protocol": "independent_eligible_sampling_v1",
        "graph_sha256": graph_fingerprint(G),
        "fact_checker_nodes": json.dumps(sorted(fact_checkers)),
        "resistant_nodes": json.dumps(sorted(resistant_nodes)),
        "initial_believer_nodes": json.dumps(sorted(initial_believers)),
        "attribute_sha256": hashlib.sha256(
            json.dumps([
                [node, adoption[node], correction[node]]
                for node in nodes
            ]).encode()
        ).hexdigest(),
        "realised_network_seed": G.graph.get(
            "realised_network_seed",
            seeds["network_seed"],
        ),
        "actual_n": len(G),
        "actual_mean_degree": (
            sum(dict(G.degree()).values()) / len(G)
        ),
        "num_fact_checkers": len(fact_checkers),
        "num_initial_believers": len(initial_believers),
        "actual_initial_believer_fraction": (
            len(initial_believers) / len(G)
        ),
        "final_reach": len(ever_believed) / len(G),
        "peak_believers": peak_believers,
        "peak_believer_fraction": peak_believers / len(G),
        "time_to_peak": time_to_peak,
        "extinction_time": extinction_time,
        "ended_with_active_believers": believer_series[-1] > 0,
        "final_believers": believer_series[-1],
        "stop_reason": (
            "time_limit"
            if believer_series[-1] > 0
            else "extinction"
        ),
    }

    return history, metrics, G