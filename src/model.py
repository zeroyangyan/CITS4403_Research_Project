from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple
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
    network_type: str = "small_world"
    ws_rewire_p: float = 0.10

    misinformation_prob: float = 0.15
    correction_prob: float = 0.35
    resistant_fraction: float = 0.05

    intervention: str = "none"
    fact_checker_fraction: float = 0.05
    initial_believers: int = 5

    # homogeneous | low | high
    heterogeneity: str = "homogeneous"

    corrected_share: bool = True
    max_steps: int = 150
    seed: int = 1


def _validate_config(cfg: ModelConfig) -> None:
    if cfg.n < 10:
        raise ValueError("n must be at least 10")

    if not 2 <= cfg.mean_degree < cfg.n:
        raise ValueError("mean_degree must be between 2 and n-1")

    for name in (
        "misinformation_prob",
        "correction_prob",
        "ws_rewire_p",
    ):
        if not 0 <= getattr(cfg, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    for name in ("resistant_fraction", "fact_checker_fraction"):
        if not 0 <= getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be in [0, 1)")

    if cfg.initial_believers < 1:
        raise ValueError("initial_believers must be positive")

    if cfg.max_steps < 1:
        raise ValueError("max_steps must be positive")

    if cfg.network_type not in (
        "random", "small_world", "scale_free"
    ):
        raise ValueError("Unknown network type")

    if cfg.intervention not in ("none", "random", "centrality"):
        raise ValueError("Unknown intervention")

    if cfg.heterogeneity not in ("homogeneous", "low", "high"):
        raise ValueError("Unknown heterogeneity level")


def generate_network(cfg: ModelConfig) -> nx.Graph:
    _validate_config(cfg)

    if cfg.network_type == "random":
        p = cfg.mean_degree / (cfg.n - 1)

        # 保留所有节点，包括可能存在的孤立节点。
        # 避免不同网络条件的实际人数发生变化。
        return nx.erdos_renyi_graph(cfg.n, p, seed=cfg.seed)

    if cfg.network_type == "small_world":
        k = cfg.mean_degree
        if k % 2:
            k += 1

        largest_even = cfg.n - 1
        if largest_even % 2:
            largest_even -= 1

        k = min(k, largest_even)

        return nx.watts_strogatz_graph(
            cfg.n, k, cfg.ws_rewire_p, seed=cfg.seed
        )

    m = max(1, round(cfg.mean_degree / 2))
    return nx.barabasi_albert_graph(cfg.n, m, seed=cfg.seed)


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
        return set(rng.sample(nodes, count))

    # centrality 在这个项目中具体指 degree-based selection。
    ranked = sorted(G.degree, key=lambda item: (-item[1], item[0]))
    return {node for node, degree in ranked[:count]}


def make_probabilities(
    nodes: List[int],
    mean: float,
    level: str,
    rng: random.Random,
) -> Dict[int, float]:
    """
    为每个用户生成固定的行为概率。

    对称生成 mean-d 和 mean+d，使实际平均值保持为 mean。
    随机打乱后分配给节点，避免按节点编号分配行为特征。
    两类行为概率分别生成，不预设它们之间的相关关系。
    """
    strengths = {
        "homogeneous": 0.0,
        "low": 0.25,
        "high": 0.90,
    }

    width = strengths[level] * min(mean, 1.0 - mean)
    values = []

    for _ in range(len(nodes) // 2):
        difference = rng.uniform(0.0, width)
        values.extend([mean - difference, mean + difference])

    if len(nodes) % 2:
        values.append(mean)

    rng.shuffle(values)
    return dict(zip(nodes, values))


def initialise_states(
    G: nx.Graph,
    cfg: ModelConfig,
    fact_checkers: Set[int],
    rng: random.Random,
) -> Tuple[Dict[int, str], Set[int]]:
    states = {node: UNAWARE for node in G.nodes()}

    for node in fact_checkers:
        states[node] = CORRECTED

    candidates = sorted(set(G.nodes()) - fact_checkers)
    resistant_count = round(len(G) * cfg.resistant_fraction)

    if resistant_count + cfg.initial_believers > len(candidates):
        raise ValueError("Not enough eligible nodes for initial states")

    resistant_nodes = set(rng.sample(candidates, resistant_count))

    for node in resistant_nodes:
        states[node] = RESISTANT

    eligible = sorted(set(candidates) - resistant_nodes)
    initial = set(rng.sample(eligible, cfg.initial_believers))

    for node in initial:
        states[node] = BELIEVER

    return states, initial


def _prob_at_least_one_success(p: float, exposures: int) -> float:
    return 1.0 - (1.0 - p) ** exposures


def run_simulation(cfg: ModelConfig):
    _validate_config(cfg)

    # 不同用途使用不同随机流。
    # 这里没有强制不同策略共享初始相信者，因此不是配对设计。
    streams = np.random.SeedSequence(cfg.seed).spawn(4)
    seeds = [int(s.generate_state(1)[0]) for s in streams]

    selection_rng = random.Random(seeds[0])
    initial_rng = random.Random(seeds[1])
    behaviour_rng = random.Random(seeds[2])
    dynamics_rng = random.Random(seeds[3])

    G = generate_network(cfg)
    nodes = sorted(G.nodes())

    fact_checkers = select_fact_checkers(
        G, cfg.intervention, cfg.fact_checker_fraction, selection_rng
    )

    adoption = make_probabilities(
        nodes, cfg.misinformation_prob,
        cfg.heterogeneity, behaviour_rng
    )
    correction = make_probabilities(
        nodes, cfg.correction_prob,
        cfg.heterogeneity, behaviour_rng
    )

    states, initial = initialise_states(
        G, cfg, fact_checkers, initial_rng
    )

    ever_believed = set(initial)
    history = []

    def record(step: int) -> None:
        history.append({
            "step": step,
            "unaware": sum(s == UNAWARE for s in states.values()),
            "believer": sum(s == BELIEVER for s in states.values()),
            "corrected": sum(s == CORRECTED for s in states.values()),
            "resistant": sum(s == RESISTANT for s in states.values()),
            "ever_believed": len(ever_believed),
        })

    record(0)

    for step in range(1, cfg.max_steps + 1):
        believers = {
            node for node, state in states.items()
            if state == BELIEVER
        }

        if not believers:
            break

        misinformation_exposures = {}
        correction_exposures = {}

        for source in sorted(believers):
            for neighbour in sorted(G.neighbors(source)):
                if states[neighbour] == UNAWARE:
                    misinformation_exposures[neighbour] = (
                        misinformation_exposures.get(neighbour, 0) + 1
                    )

        correction_sources = set(fact_checkers)

        if cfg.corrected_share:
            correction_sources.update(
                node for node, state in states.items()
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

        for node in sorted(misinformation_exposures):
            p = _prob_at_least_one_success(
                adoption[node], misinformation_exposures[node]
            )
            if dynamics_rng.random() < p:
                to_believer.add(node)

        for node in sorted(correction_exposures):
            p = _prob_at_least_one_success(
                correction[node], correction_exposures[node]
            )
            if dynamics_rng.random() < p:
                to_corrected.add(node)

        # 同步更新：本步新相信的用户从下一步开始传播。
        for node in to_believer:
            states[node] = BELIEVER
            ever_believed.add(node)

        for node in to_corrected:
            states[node] = CORRECTED

        record(step)

    believer_series = [row["believer"] for row in history]
    peak = max(believer_series)
    active_at_end = believer_series[-1] > 0

    metrics = {
        **asdict(cfg),
        "actual_n": len(G),
        "actual_mean_degree": 2 * G.number_of_edges() / len(G),
        "num_fact_checkers": len(fact_checkers),
        "adoption_mean": float(np.mean(list(adoption.values()))),
        "adoption_std": float(np.std(list(adoption.values()))),
        "correction_mean": float(np.mean(list(correction.values()))),
        "correction_std": float(np.std(list(correction.values()))),
        "final_reach": len(ever_believed) / len(G),
        "peak_believers": peak,
        "peak_believer_fraction": peak / len(G),
        # 若存在多个相同峰值，取第一次达到峰值的时间。
        "time_to_peak": history[believer_series.index(peak)]["step"],
        "final_believers": believer_series[-1],
        "ended_with_active_believers": active_at_end,
        "extinction_time": (
            None if active_at_end else history[-1]["step"]
        ),
        "stop_reason": (
            "time_limit" if active_at_end else "extinction"
        ),
        "selection_seed": seeds[0],
        "initialisation_seed": seeds[1],
        "behaviour_seed": seeds[2],
        "dynamics_seed": seeds[3],
    }

    return history, metrics, G