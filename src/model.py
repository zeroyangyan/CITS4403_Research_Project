from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import random

import networkx as nx
import numpy as np


UNAWARE = "Unaware"
BELIEVER = "Believer"
CORRECTED = "Corrected"
RESISTANT = "Resistant"

ORDINARY_ROLE = "ordinary"
FACT_CHECKER_ROLE = "fact_checker"
RESISTANT_ROLE = "resistant"


@dataclass
class ModelConfig:
    """All parameters needed for one reproducible simulation run."""

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

    # Latest-main experiment interface.
    # homogeneous | low | high
    heterogeneity: str = "homogeneous"

    # Feature-branch compatibility / explicit Beta mode.
    # None: use `heterogeneity` above.
    # homogeneous: force baseline probabilities.
    # heterogeneous: sample Beta-distributed probabilities once per ordinary agent.
    model_mode: Optional[str] = None
    heterogeneity_distribution: str = "beta"
    susceptibility_concentration: float = 20.0
    correction_concentration: float = 20.0

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

    if cfg.network_type not in ("random", "small_world", "scale_free"):
        raise ValueError("Unknown network type")

    if cfg.intervention not in ("none", "random", "centrality"):
        raise ValueError("Unknown intervention")

    if cfg.heterogeneity not in ("homogeneous", "low", "high"):
        raise ValueError("Unknown heterogeneity level")

    if cfg.model_mode not in (None, "homogeneous", "heterogeneous"):
        raise ValueError(
            "model_mode must be None, 'homogeneous', or 'heterogeneous'"
        )

    if cfg.heterogeneity_distribution != "beta":
        raise ValueError(
            "heterogeneity_distribution currently supports only 'beta'"
        )

    if cfg.susceptibility_concentration <= 0:
        raise ValueError("susceptibility_concentration must be > 0")

    if cfg.correction_concentration <= 0:
        raise ValueError("correction_concentration must be > 0")


def generate_network(cfg: ModelConfig) -> nx.Graph:
    """Generate a network while preserving all configured nodes."""
    _validate_config(cfg)

    if cfg.network_type == "random":
        p = cfg.mean_degree / (cfg.n - 1)
        # Preserve all nodes, including possible isolates, so n is comparable
        # across experimental conditions.
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
    m = min(m, cfg.n - 1)
    return nx.barabasi_albert_graph(cfg.n, m, seed=cfg.seed)


def select_fact_checkers(
    G: nx.Graph,
    strategy: str,
    fraction: float,
    rng: random.Random,
) -> Set[int]:
    """Select a fixed set of fact-checkers under the requested strategy."""
    if strategy == "none" or fraction <= 0:
        return set()

    count = max(1, round(len(G) * fraction))
    nodes = sorted(G.nodes())

    if strategy == "random":
        return set(rng.sample(nodes, min(count, len(nodes))))

    if strategy == "centrality":
        ranked = sorted(G.degree, key=lambda item: (-item[1], item[0]))
        return {node for node, _degree in ranked[:count]}

    raise ValueError(f"Unknown intervention strategy: {strategy}")


def make_probabilities(
    nodes: List[int],
    mean: float,
    level: str,
    rng: random.Random,
) -> Dict[int, float]:
    """
    Generate fixed probabilities for the latest-main low/high experiment.

    Values are generated symmetrically around the configured mean so that the
    realised mean remains equal to the baseline probability.
    """
    strengths = {
        "homogeneous": 0.0,
        "low": 0.25,
        "high": 0.90,
    }

    width = strengths[level] * min(mean, 1.0 - mean)
    values: List[float] = []

    for _ in range(len(nodes) // 2):
        difference = rng.uniform(0.0, width)
        values.extend([mean - difference, mean + difference])

    if len(nodes) % 2:
        values.append(mean)

    rng.shuffle(values)
    return dict(zip(nodes, values))


def _sample_beta_with_mean(
    rng: np.random.Generator,
    mean: float,
    concentration: float,
) -> float:
    """Sample Beta(mean*k, (1-mean)*k), handling boundary means exactly."""
    if mean <= 0.0:
        return 0.0
    if mean >= 1.0:
        return 1.0

    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    return float(np.clip(rng.beta(alpha, beta), 0.0, 1.0))


def _prob_at_least_one_success(p: float, exposures: int) -> float:
    """Probability of at least one success across k independent exposures."""
    if exposures <= 0:
        return 0.0
    return 1.0 - (1.0 - p) ** exposures


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


def initialise_agent_attributes(
    G: nx.Graph,
    cfg: ModelConfig,
    fact_checkers: Set[int],
    states: Dict[int, str],
    behaviour_rng: random.Random,
    beta_rng: np.random.Generator,
) -> Tuple[Dict[int, float], Dict[int, float], str, str]:
    """
    Assign fixed behavioural probabilities once during initialisation.

    Two interfaces are retained so the feature branch and latest main can merge
    without discarding either experiment design:

    * model_mode is None:
      use latest-main `heterogeneity` levels (homogeneous/low/high).
    * model_mode == "homogeneous":
      force the original homogeneous baseline.
    * model_mode == "heterogeneous":
      use the feature branch's documented Beta distribution.

    Fact-checkers and resistant agents keep their protected state behaviour.
    Their stored probability attributes are bounded baseline values for
    inspection/export only.
    """
    nodes = sorted(G.nodes())

    if cfg.model_mode == "heterogeneous":
        adoption = {}
        correction = {}
        for node in nodes:
            adoption[node] = _sample_beta_with_mean(
                beta_rng,
                cfg.misinformation_prob,
                cfg.susceptibility_concentration,
            )
            correction[node] = _sample_beta_with_mean(
                beta_rng,
                cfg.correction_prob,
                cfg.correction_concentration,
            )
        effective_mode = "heterogeneous"
        effective_distribution = cfg.heterogeneity_distribution

    elif cfg.model_mode == "homogeneous":
        adoption = {node: cfg.misinformation_prob for node in nodes}
        correction = {node: cfg.correction_prob for node in nodes}
        effective_mode = "homogeneous"
        effective_distribution = "constant"

    else:
        adoption = make_probabilities(
            nodes, cfg.misinformation_prob, cfg.heterogeneity, behaviour_rng
        )
        correction = make_probabilities(
            nodes, cfg.correction_prob, cfg.heterogeneity, behaviour_rng
        )
        effective_mode = (
            "homogeneous" if cfg.heterogeneity == "homogeneous"
            else "heterogeneous"
        )
        effective_distribution = (
            "constant" if cfg.heterogeneity == "homogeneous"
            else "symmetric_uniform"
        )

    for node in nodes:
        if node in fact_checkers:
            role = FACT_CHECKER_ROLE
        elif states[node] == RESISTANT:
            role = RESISTANT_ROLE
        else:
            role = ORDINARY_ROLE

        G.nodes[node]["agent_role"] = role
        G.nodes[node]["initial_state"] = states[node]

        # Protected nodes never adopt misinformation, but keeping bounded values
        # on every node makes validation/export straightforward.
        if role == ORDINARY_ROLE:
            susceptibility = float(np.clip(adoption[node], 0.0, 1.0))
            receptiveness = float(np.clip(correction[node], 0.0, 1.0))
        else:
            susceptibility = cfg.misinformation_prob
            receptiveness = cfg.correction_prob

        G.nodes[node]["misinformation_susceptibility"] = susceptibility
        G.nodes[node]["correction_receptiveness"] = receptiveness

    return adoption, correction, effective_mode, effective_distribution


def _ordinary_attribute_summary(G: nx.Graph) -> Dict[str, float]:
    ordinary = [
        node
        for node in G.nodes()
        if G.nodes[node].get("agent_role") == ORDINARY_ROLE
    ]

    if not ordinary:
        return {
            "realised_susceptibility_mean": float("nan"),
            "realised_susceptibility_std": float("nan"),
            "realised_correction_receptiveness_mean": float("nan"),
            "realised_correction_receptiveness_std": float("nan"),
        }

    susceptibility = np.array(
        [
            G.nodes[node]["misinformation_susceptibility"]
            for node in ordinary
        ],
        dtype=float,
    )
    receptiveness = np.array(
        [
            G.nodes[node]["correction_receptiveness"]
            for node in ordinary
        ],
        dtype=float,
    )

    return {
        "realised_susceptibility_mean": float(susceptibility.mean()),
        "realised_susceptibility_std": float(susceptibility.std(ddof=0)),
        "realised_correction_receptiveness_mean": float(receptiveness.mean()),
        "realised_correction_receptiveness_std": float(
            receptiveness.std(ddof=0)
        ),
    }


def run_simulation(
    cfg: ModelConfig,
) -> Tuple[List[dict], dict, nx.Graph]:
    _validate_config(cfg)

    # Explicit feature-branch mode preserves the original baseline RNG sequence.
    # This keeps homogeneous regression checks stable while still allowing the
    # latest-main factorial design to use separated random streams when
    # model_mode is left as None.
    if cfg.model_mode is not None:
        baseline_rng = random.Random(cfg.seed)
        selection_rng = baseline_rng
        initial_rng = baseline_rng
        behaviour_rng = random.Random(cfg.seed + 20_000_033)
        dynamics_rng = baseline_rng
        beta_seed = cfg.seed + 10_000_019
        beta_rng = np.random.default_rng(beta_seed)
        seed_record = {
            "selection_seed": cfg.seed,
            "initialisation_seed": cfg.seed,
            "behaviour_seed": cfg.seed + 20_000_033,
            "dynamics_seed": cfg.seed,
            "attribute_seed": beta_seed,
        }
    else:
        # Latest-main design: independent streams for selection, initial state,
        # behaviour attributes, dynamics, and Beta attributes.
        streams = np.random.SeedSequence(cfg.seed).spawn(5)
        seeds = [int(stream.generate_state(1)[0]) for stream in streams]

        selection_rng = random.Random(seeds[0])
        initial_rng = random.Random(seeds[1])
        behaviour_rng = random.Random(seeds[2])
        dynamics_rng = random.Random(seeds[3])
        beta_rng = np.random.default_rng(seeds[4])
        seed_record = {
            "selection_seed": seeds[0],
            "initialisation_seed": seeds[1],
            "behaviour_seed": seeds[2],
            "dynamics_seed": seeds[3],
            "attribute_seed": seeds[4],
        }

    G = generate_network(cfg)

    fact_checkers = select_fact_checkers(
        G,
        cfg.intervention,
        cfg.fact_checker_fraction,
        selection_rng,
    )

    states, initial = initialise_states(
        G,
        cfg,
        fact_checkers,
        initial_rng,
    )

    (
        _adoption,
        _correction,
        effective_mode,
        effective_distribution,
    ) = initialise_agent_attributes(
        G,
        cfg,
        fact_checkers,
        states,
        behaviour_rng,
        beta_rng,
    )

    ever_believed = set(initial)
    history: List[dict] = []

    def record(step: int) -> None:
        history.append(
            {
                "step": step,
                "unaware": sum(
                    state == UNAWARE for state in states.values()
                ),
                "believer": sum(
                    state == BELIEVER for state in states.values()
                ),
                "corrected": sum(
                    state == CORRECTED for state in states.values()
                ),
                "resistant": sum(
                    state == RESISTANT for state in states.values()
                ),
                "ever_believed": len(ever_believed),
            }
        )

    record(0)

    for step in range(1, cfg.max_steps + 1):
        believers = {
            node
            for node, state in states.items()
            if state == BELIEVER
        }

        if not believers:
            break

        misinformation_exposures: Dict[int, int] = {}
        correction_exposures: Dict[int, int] = {}

        # Preserve deterministic iteration in latest-main mode. In explicit
        # feature mode, insertion order is already stable for integer node IDs.
        source_nodes = (
            believers
            if cfg.model_mode is not None
            else sorted(believers)
        )
        for source in source_nodes:
            neighbours = (
                G.neighbors(source)
                if cfg.model_mode is not None
                else sorted(G.neighbors(source))
            )
            for neighbour in neighbours:
                if states[neighbour] == UNAWARE:
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

        correction_source_nodes = (
            correction_sources
            if cfg.model_mode is not None
            else sorted(correction_sources)
        )
        for source in correction_source_nodes:
            neighbours = (
                G.neighbors(source)
                if cfg.model_mode is not None
                else sorted(G.neighbors(source))
            )
            for neighbour in neighbours:
                if states[neighbour] == BELIEVER:
                    correction_exposures[neighbour] = (
                        correction_exposures.get(neighbour, 0) + 1
                    )

        to_believer: Set[int] = set()
        to_corrected: Set[int] = set()

        misinformation_nodes = (
            misinformation_exposures.items()
            if cfg.model_mode is not None
            else (
                (node, misinformation_exposures[node])
                for node in sorted(misinformation_exposures)
            )
        )
        for node, exposures in misinformation_nodes:
            single_prob = G.nodes[node]["misinformation_susceptibility"]
            p = _prob_at_least_one_success(single_prob, exposures)
            if dynamics_rng.random() < p:
                to_believer.add(node)

        correction_nodes = (
            correction_exposures.items()
            if cfg.model_mode is not None
            else (
                (node, correction_exposures[node])
                for node in sorted(correction_exposures)
            )
        )
        for node, exposures in correction_nodes:
            single_prob = G.nodes[node]["correction_receptiveness"]
            p = _prob_at_least_one_success(single_prob, exposures)
            if dynamics_rng.random() < p:
                to_corrected.add(node)

        for node in to_believer:
            states[node] = BELIEVER
            ever_believed.add(node)

        for node in to_corrected:
            states[node] = CORRECTED

        record(step)

    believer_series = [row["believer"] for row in history]
    peak = max(believer_series)
    active_at_end = believer_series[-1] > 0

    for node in G.nodes():
        G.nodes[node]["final_state"] = states[node]

    ordinary_summary = _ordinary_attribute_summary(G)
    ordinary_nodes = [
        node
        for node in G.nodes()
        if G.nodes[node]["agent_role"] == ORDINARY_ROLE
    ]

    metrics = {
        **asdict(cfg),
        "model_mode": effective_mode,
        "effective_heterogeneity_distribution": effective_distribution,
        "actual_n": len(G),
        "actual_mean_degree": (
            2 * G.number_of_edges() / len(G) if len(G) else 0.0
        ),
        "num_fact_checkers": len(fact_checkers),
        "adoption_mean": float(
            np.mean(
                [
                    G.nodes[node]["misinformation_susceptibility"]
                    for node in ordinary_nodes
                ]
            )
        ),
        "adoption_std": ordinary_summary[
            "realised_susceptibility_std"
        ],
        "correction_mean": float(
            np.mean(
                [
                    G.nodes[node]["correction_receptiveness"]
                    for node in ordinary_nodes
                ]
            )
        ),
        "correction_std": ordinary_summary[
            "realised_correction_receptiveness_std"
        ],
        "final_reach": len(ever_believed) / len(G),
        "peak_believers": peak,
        "peak_believer_fraction": peak / len(G),
        "time_to_peak": history[believer_series.index(peak)]["step"],
        "final_believers": believer_series[-1],
        "ended_with_active_believers": active_at_end,
        "extinction_time": (
            None if active_at_end else history[-1]["step"]
        ),
        "stop_reason": (
            "time_limit" if active_at_end else "extinction"
        ),
        **seed_record,
        **ordinary_summary,
    }

    return history, metrics, G
