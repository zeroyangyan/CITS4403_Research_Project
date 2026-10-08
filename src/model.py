from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
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


HETEROGENEITY_STRENGTH = {
    "homogeneous": 0.0,
    "low": 0.25,
    "high": 0.90,
}


def _validate_config(cfg):
    if cfg.n < 10:
        raise ValueError("n must be at least 10")

    if not 2 <= cfg.mean_degree < cfg.n:
        raise ValueError("mean_degree must be between 2 and n-1")

    if cfg.network_type not in (
        "random", "small_world", "scale_free"
    ):
        raise ValueError("Unknown network type")

    if cfg.intervention not in (
        "none", "random", "centrality"
    ):
        raise ValueError("Unknown intervention")

    if cfg.heterogeneity not in HETEROGENEITY_STRENGTH:
        raise ValueError("Unknown heterogeneity level")

    for name in (
        "misinformation_prob",
        "correction_prob",
        "ws_rewire_p",
    ):
        if not 0 <= getattr(cfg, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    for name in (
        "resistant_fraction",
        "fact_checker_fraction",
    ):
        if not 0 <= getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be in [0, 1)")

    if cfg.initial_believers < 1:
        raise ValueError("initial_believers must be positive")

    if cfg.max_steps < 1 or cfg.seed < 0:
        raise ValueError(
            "max_steps must be positive and seed nonnegative"
        )


def resolve_seeds(cfg):
    """
    网络和属性种子不依赖策略。
    核查者选择、初始化和状态转移使用各自独立随机流。
    """
    strategy_id = {
        "none": 0,
        "random": 1,
        "centrality": 2,
    }[cfg.intervention]

    def derive(channel, strategy=0):
        sequence = np.random.SeedSequence(
            [cfg.seed, channel, strategy]
        )
        return int(sequence.generate_state(1)[0])

    return {
        "network_seed": derive(0),
        "attribute_seed": derive(1),
        "selection_seed": derive(2, strategy_id),
        "initialisation_seed": derive(3, strategy_id),
        "transition_seed": derive(4, strategy_id),
    }


def fingerprint(payload):
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def graph_fingerprint(G):
    return fingerprint({
        "nodes": sorted(G.nodes()),
        "edges": sorted(
            (min(a, b), max(a, b))
            for a, b in G.edges()
        ),
    })


def generate_network(cfg):
    _validate_config(cfg)
    network_seed = resolve_seeds(cfg)["network_seed"]

    if cfg.network_type == "random":
        p = cfg.mean_degree / (cfg.n - 1)

        # 保留原来的连通性处理方式。
        for offset in range(20):
            G = nx.erdos_renyi_graph(
                cfg.n,
                p,
                seed=network_seed + offset,
            )
            if nx.is_connected(G):
                return G

        G = nx.erdos_renyi_graph(
            cfg.n, p, seed=network_seed
        )
        largest = max(
            nx.connected_components(G), key=len
        )
        G = G.subgraph(largest).copy()
        return nx.convert_node_labels_to_integers(G)

    if cfg.network_type == "small_world":
        k = cfg.mean_degree
        if k % 2:
            k += 1

        largest_even = cfg.n - 1
        if largest_even % 2:
            largest_even -= 1

        k = min(k, largest_even)

        return nx.watts_strogatz_graph(
            cfg.n,
            k,
            cfg.ws_rewire_p,
            seed=network_seed,
        )

    m = max(1, round(cfg.mean_degree / 2))
    m = min(m, cfg.n - 1)

    return nx.barabasi_albert_graph(
        cfg.n, m, seed=network_seed
    )


def make_probabilities(nodes, mean, level, rng):
    """
    分布规则：
    d ~ Uniform(0, strength * min(mean, 1-mean))
    成对生成 mean-d 和 mean+d，再随机分配给节点。
    奇数个节点时，额外节点取 mean。

    保证全体节点的实际平均值等于配置平均值，
    并保证概率在 [0, 1] 内。

    各节点属性在一次模拟中固定。
    """
    strength = HETEROGENEITY_STRENGTH[level]
    width = strength * min(mean, 1.0 - mean)

    if width == 0:
        return {node: mean for node in nodes}

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


def select_fact_checkers(G, strategy, fraction, rng):
    if strategy == "none" or fraction <= 0:
        return set()

    count = max(1, round(len(G) * fraction))
    nodes = sorted(G.nodes())

    if strategy == "random":
        return set(rng.sample(nodes, count))

    # centrality 在本项目中表示 degree-based selection。
    ranked = sorted(
        G.degree,
        key=lambda item: (-item[1], item[0]),
    )

    return {node for node, degree in ranked[:count]}


def initialise_states(G, cfg, fact_checkers, rng):
    """
    核查者、抵抗者、初始相信者互不重叠。
    初始状态按策略独立抽样，不保证节点身份相同。
    """
    states = {node: UNAWARE for node in G.nodes()}

    for node in fact_checkers:
        states[node] = CORRECTED

    candidates = sorted(
        set(G.nodes()) - fact_checkers
    )
    resistant_count = round(
        len(G) * cfg.resistant_fraction
    )

    if (
        resistant_count + cfg.initial_believers
        > len(candidates)
    ):
        raise ValueError(
            "Not enough eligible nodes for initial states"
        )

    resistant_nodes = set(
        rng.sample(candidates, resistant_count)
    )

    for node in resistant_nodes:
        states[node] = RESISTANT

    eligible = sorted(
        set(candidates) - resistant_nodes
    )
    initial = set(
        rng.sample(eligible, cfg.initial_believers)
    )

    for node in initial:
        states[node] = BELIEVER

    return states, initial


def _prob_at_least_one_success(p, exposures):
    if exposures <= 0:
        return 0.0
    return 1.0 - (1.0 - p) ** exposures


def run_simulation(cfg):
    _validate_config(cfg)
    seeds = resolve_seeds(cfg)

    attribute_rng = random.Random(
        seeds["attribute_seed"]
    )
    selection_rng = random.Random(
        seeds["selection_seed"]
    )
    initial_rng = random.Random(
        seeds["initialisation_seed"]
    )
    transition_rng = random.Random(
        seeds["transition_seed"]
    )

    G = generate_network(cfg)
    nodes = sorted(G.nodes())

    # 先为所有节点分配潜在属性，避免策略改变属性分配。
    # 核查者与抵抗者仍遵守原来的保护规则。
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

    states, initial = initialise_states(
        G, cfg, fact_checkers, initial_rng
    )

    resistant_nodes = {
        node for node, state in states.items()
        if state == RESISTANT
    }

    ordinary_nodes = sorted(
        set(nodes) - fact_checkers - resistant_nodes
    )

    ever_believed = set(initial)
    history = []

    def record(step):
        history.append({
            "step": step,
            "unaware": sum(
                state == UNAWARE
                for state in states.values()
            ),
            "believer": sum(
                state == BELIEVER
                for state in states.values()
            ),
            "corrected": sum(
                state == CORRECTED
                for state in states.values()
            ),
            "resistant": sum(
                state == RESISTANT
                for state in states.values()
            ),
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
                        misinformation_exposures.get(
                            neighbour, 0
                        ) + 1
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
                        correction_exposures.get(
                            neighbour, 0
                        ) + 1
                    )

        to_believer = set()
        to_corrected = set()

        for node, exposures in sorted(
            misinformation_exposures.items()
        ):
            # 使用接收用户的相信概率。
            p = _prob_at_least_one_success(
                adoption[node], exposures
            )
            if transition_rng.random() < p:
                to_believer.add(node)

        for node, exposures in sorted(
            correction_exposures.items()
        ):
            # 使用接收用户的接受纠正概率。
            p = _prob_at_least_one_success(
                correction[node], exposures
            )
            if transition_rng.random() < p:
                to_corrected.add(node)

        # 同步更新；新相信者从下一步开始传播。
        for node in to_believer:
            states[node] = BELIEVER
            ever_believed.add(node)

        for node in to_corrected:
            states[node] = CORRECTED

        record(step)

    believer_series = [
        row["believer"] for row in history
    ]
    peak = max(believer_series)
    peak_index = believer_series.index(peak)
    active_at_end = believer_series[-1] > 0

    metrics = {
        **asdict(cfg),
        **seeds,
        "initialisation_protocol":
            "independent_eligible_sampling_v1",
        "heterogeneity_distribution":
            "paired_symmetric_uniform",
        "heterogeneity_strength":
            HETEROGENEITY_STRENGTH[cfg.heterogeneity],
        "graph_sha256": graph_fingerprint(G),
        "attribute_sha256": fingerprint([
            [node, adoption[node], correction[node]]
            for node in nodes
        ]),
        "actual_n": len(G),
        "actual_mean_degree":
            2 * G.number_of_edges() / len(G),
        "num_fact_checkers": len(fact_checkers),
        "fact_checker_nodes":
            json.dumps(sorted(fact_checkers)),
        "resistant_nodes":
            json.dumps(sorted(resistant_nodes)),
        "initial_believer_nodes":
            json.dumps(sorted(initial)),
        "adoption_mean": float(np.mean([
            adoption[node] for node in nodes
        ])),
        "adoption_std": float(np.std([
            adoption[node] for node in nodes
        ])),
        "correction_mean": float(np.mean([
            correction[node] for node in nodes
        ])),
        "correction_std": float(np.std([
            correction[node] for node in nodes
        ])),
        # 普通用户子集的均值可能因角色选择略有变化，
        # 所以额外记录，方便检查策略间的属性构成。
        "ordinary_adoption_mean": float(np.mean([
            adoption[node] for node in ordinary_nodes
        ])),
        "ordinary_correction_mean": float(np.mean([
            correction[node] for node in ordinary_nodes
        ])),
        "final_reach": len(ever_believed) / len(G),
        "peak_believers": peak,
        "peak_believer_fraction": peak / len(G),
        "time_to_peak": history[peak_index]["step"],
        "final_believers": believer_series[-1],
        "ended_with_active_believers": active_at_end,
        "extinction_time": (
            None if active_at_end
            else history[-1]["step"]
        ),
        "stop_reason": (
            "time_limit" if active_at_end
            else "extinction"
        ),
    }

    return history, metrics, G