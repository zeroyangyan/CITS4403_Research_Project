from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
import random

import networkx as nx
import numpy as np


# 四种用户状态。使用常量可以避免在代码中重复输入而产生拼写错误。
UNAWARE = "Unaware"
BELIEVER = "Believer"
CORRECTED = "Corrected"
RESISTANT = "Resistant"


@dataclass
class ModelConfig:
    """保存一次仿真的全部参数，方便重复实验和结果复现。"""

    # 网络规模与结构参数
    n: int = 500
    mean_degree: int = 8
    network_type: str = "small_world"   # random | small_world | scale_free
    ws_rewire_p: float = 0.10

    # 信息传播参数
    misinformation_prob: float = 0.15
    correction_prob: float = 0.35
    resistant_fraction: float = 0.05

    # 干预策略参数
    intervention: str = "none"          # none | random | centrality
    fact_checker_fraction: float = 0.05
    initial_believers: int = 5

    corrected_share: bool = True# 表示被纠正的普通用户也会继续传播纠正信息
    max_steps: int = 150
    seed: int = 1


def _validate_config(cfg: ModelConfig) -> None:
    """在运行前检查参数范围，避免生成无效网络或无意义的概率。"""
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
    """根据配置生成网络，并让不同网络的平均度尽量接近。"""
    _validate_config(cfg)
    rng_seed = cfg.seed

    if cfg.network_type == "random":
        # Erdős-Rényi 随机网络：每一对节点以概率 p 建立连接。
        # p=平均度/(节点数-1)，因此期望平均度约等于 mean_degree。
        p = cfg.mean_degree / (cfg.n - 1)
        # 尝试不同种子寻找连通网络；若仍不连通，则只保留最大连通分量。
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
            # 重新编号，使节点编号连续（0, 1, 2, ...）。
            G = nx.convert_node_labels_to_integers(G)

    elif cfg.network_type == "small_world":
        # Watts-Strogatz 小世界网络要求 k 为偶数。
        k = cfg.mean_degree if cfg.mean_degree % 2 == 0 else cfg.mean_degree + 1
        k = min(k, cfg.n - 1 - ((cfg.n - 1) % 2))
        G = nx.watts_strogatz_graph(cfg.n, k, cfg.ws_rewire_p, seed=rng_seed)# ws_rewire_p 控制边被随机重连的概率。

    elif cfg.network_type == "scale_free":
        # Barabási-Albert 无标度网络：每个新节点连接到 m 个已有节点。
        # 该模型的平均度约为 2m，所以令 m≈mean_degree/2。
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
    """按照指定策略选择一组固定的事实核查者。"""
    if strategy == "none" or fraction <= 0:
        return set()

    # 至少选择1人，并把比例换算为实际人数。
    count = max(1, round(len(G) * fraction))
    nodes = list(G.nodes())

    if strategy == "random":
        # 随机策略：所有节点被选中的机会相同。
        return set(rng.sample(nodes, min(count, len(nodes))))

    if strategy == "centrality":
        # 中心性策略：按照连接数（degree）从高到低排序。
        # 连接数相同时按节点编号排序，使结果稳定且可复现。
        ranked = sorted(G.degree, key=lambda x: (-x[1], x[0]))
        return {node for node, _ in ranked[:count]}

    raise ValueError(f"Unknown intervention strategy: {strategy}")


def _prob_at_least_one_success(single_prob: float, exposures: int) -> float:
    """计算多次独立接触中至少成功一次的概率：1-(1-p)^k。"""
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
    初始化全部用户的状态。

    事实核查者从正确信息状态开始，并且不会成为相信的人；
    抵抗者也不会相信错误信息。
    """
    # 所有用户最初均设为尚未接触错误信息。
    states = {node: UNAWARE for node in G.nodes()}

    # 先安排事实核查者，再从剩余用户中选择抵抗者和初始相信者。
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
    运行一次同步更新的错误信息传播仿真。

    每个时间步的顺序：
    1）当前的错误信息相信者向相邻的无意识用户传播错误信息；
    2）事实核查者和（可选的正确信息用户向错误信息相信者传播纠正信息；
    3）统一应用本时间步产生的状态变化。

    返回值：每一步的历史记录、最终指标字典，以及生成的网络 G。
    """
    # 固定 Python 和 NumPy 的随机种子，使相同配置能够复现。
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    # 建立网络、选择事实核查者，并初始化用户状态。
    G = generate_network(cfg)
    fact_checkers = select_fact_checkers(
        G, cfg.intervention, cfg.fact_checker_fraction, rng
    )
    states, initial_believers = initialise_states(G, cfg, fact_checkers, rng)

    ever_believed = set(initial_believers)# ever_believed 用于统计曾经相信过错误信息的所有用户。
    history: List[dict] = []

    def record(step: int) -> None:
        """统计当前各状态人数，并保存到时间序列 history"""
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

    # 第0步表示传播开始前的初始状态
    record(0)

    for step in range(1, cfg.max_steps + 1):
        believers = {n for n, s in states.items() if s == BELIEVER}
        # 如果已经没有相信者，错误信息传播结束，可提前停止循环。
        if not believers:
            break

        # 两个字典分别记录每个用户在本时间步收到的接触次数。
        misinformation_exposures: Dict[int, int] = {}
        correction_exposures: Dict[int, int] = {}

        # 统计错误信息接触：只有无意识用户可能在本步变成错误信息相信的人。
        for source in believers:
            for nbr in G.neighbors(source):
                if states[nbr] == UNAWARE and nbr not in fact_checkers:
                    misinformation_exposures[nbr] = misinformation_exposures.get(nbr, 0) + 1

        # 建立纠正信息来源集合。事实核查者始终属于纠正来源。
        correction_sources = set(fact_checkers)
        if cfg.corrected_share:
            # 若启用 corrected_share，所有已被纠正的用户也能传播纠正信息。
            correction_sources.update(
                n for n, s in states.items() if s == CORRECTED
            )

        for source in correction_sources:
            for nbr in G.neighbors(source):
                if states[nbr] == BELIEVER:
                    correction_exposures[nbr] = correction_exposures.get(nbr, 0) + 1

        to_believer: Set[int] = set()
        to_corrected: Set[int] = set()

        # 根据接触次数计算采纳错误信息的概率，并进行随机抽样。
        for node, exposures in misinformation_exposures.items():
            p = _prob_at_least_one_success(cfg.misinformation_prob, exposures)
            if rng.random() < p:
                to_believer.add(node)

        # 根据纠正信息接触次数，判断错误信息相信者是否会变成正确信息者。
        for node, exposures in correction_exposures.items():
            p = _prob_at_least_one_success(cfg.correction_prob, exposures)
            if rng.random() < p:
                to_corrected.add(node)

        # 同步更新：先完成所有概率判断，再统一改变状态。
        # 因此本步新产生的错误信息相信者不会立刻继续传播。
        for node in to_believer:
            if states[node] == UNAWARE:
                states[node] = BELIEVER
                ever_believed.add(node)

        for node in to_corrected:
            if states[node] == BELIEVER:
                states[node] = CORRECTED

        record(step)

    # 从历史记录中提取相信者数量，用来计算传播峰值和峰值时间。
    believer_series = [row["believer"] for row in history]
    peak_believers = max(believer_series)
    time_to_peak = believer_series.index(peak_believers)
    extinction_time: Optional[int] = None
    # 最后一步为0，表示错误信息已经停止传播。
    if believer_series[-1] == 0:
        extinction_time = history[-1]["step"]

    # 汇总配置和结果。asdict(cfg)会把配置对象转换成普通字典。
    metrics = {
        **asdict(cfg),
        "actual_n": len(G),
        "actual_mean_degree": (
            sum(dict(G.degree()).values()) / len(G) if len(G) else 0.0
        ),
        "num_fact_checkers": len(fact_checkers),
        # final_reach 是“曾经相信过”的人数比例，不只是最后一刻的比例。
        "final_reach": len(ever_believed) / len(G),
        "peak_believers": peak_believers,
        "peak_believer_fraction": peak_believers / len(G),
        "time_to_peak": time_to_peak,
        "extinction_time": extinction_time,
        "ended_with_active_believers": believer_series[-1] > 0,
    }

    return history, metrics, G
