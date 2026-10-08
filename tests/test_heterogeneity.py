import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import (  # noqa: E402
    FACT_CHECKER_ROLE,
    ORDINARY_ROLE,
    RESISTANT,
    RESISTANT_ROLE,
    ModelConfig,
    _prob_at_least_one_success,
    run_simulation,
)


class HeterogeneityTests(unittest.TestCase):
    def test_homogeneous_baseline_regression(self):
        cfg = ModelConfig(
            n=100,
            mean_degree=6,
            network_type="small_world",
            intervention="random",
            fact_checker_fraction=0.05,
            misinformation_prob=0.15,
            correction_prob=0.35,
            resistant_fraction=0.05,
            initial_believers=5,
            max_steps=40,
            seed=123,
            model_mode="homogeneous",
        )
        history, metrics, _ = run_simulation(cfg)
        self.assertEqual(metrics["actual_n"], 100)
        self.assertEqual(metrics["num_fact_checkers"], 5)
        self.assertAlmostEqual(metrics["final_reach"], 0.78)
        self.assertEqual(metrics["peak_believers"], 31)
        self.assertEqual(metrics["time_to_peak"], 8)
        self.assertEqual(metrics["extinction_time"], 15)
        self.assertFalse(metrics["ended_with_active_believers"])
        self.assertEqual(history[-1]["ever_believed"], 78)

    def test_fixed_seed_reproduces_agent_attributes(self):
        cfg = ModelConfig(
            n=80,
            mean_degree=6,
            network_type="scale_free",
            intervention="random",
            fact_checker_fraction=0.05,
            max_steps=10,
            seed=77,
            model_mode="heterogeneous",
        )
        _, _, g1 = run_simulation(cfg)
        _, _, g2 = run_simulation(cfg)
        attrs1 = [
            (
                n,
                g1.nodes[n]["agent_role"],
                g1.nodes[n]["misinformation_susceptibility"],
                g1.nodes[n]["correction_receptiveness"],
            )
            for n in sorted(g1.nodes())
        ]
        attrs2 = [
            (
                n,
                g2.nodes[n]["agent_role"],
                g2.nodes[n]["misinformation_susceptibility"],
                g2.nodes[n]["correction_receptiveness"],
            )
            for n in sorted(g2.nodes())
        ]
        self.assertEqual(attrs1, attrs2)

    def test_probability_bounds_and_multiple_exposure_rule(self):
        cfg = ModelConfig(
            n=100,
            mean_degree=8,
            network_type="random",
            intervention="random",
            fact_checker_fraction=0.05,
            seed=9,
            model_mode="heterogeneous",
        )
        _, _, graph = run_simulation(cfg)
        for node in graph.nodes():
            susceptibility = graph.nodes[node]["misinformation_susceptibility"]
            receptiveness = graph.nodes[node]["correction_receptiveness"]
            self.assertGreaterEqual(susceptibility, 0.0)
            self.assertLessEqual(susceptibility, 1.0)
            self.assertGreaterEqual(receptiveness, 0.0)
            self.assertLessEqual(receptiveness, 1.0)

        self.assertAlmostEqual(_prob_at_least_one_success(0.2, 3), 1 - 0.8**3)

    def test_state_count_conservation_and_protected_agents(self):
        cfg = ModelConfig(
            n=100,
            mean_degree=8,
            network_type="scale_free",
            intervention="centrality",
            fact_checker_fraction=0.10,
            resistant_fraction=0.10,
            misinformation_prob=1.0,
            correction_prob=1.0,
            initial_believers=5,
            max_steps=20,
            seed=45,
            model_mode="heterogeneous",
        )
        history, _, graph = run_simulation(cfg)

        for row in history:
            total = row["unaware"] + row["believer"] + row["corrected"] + row["resistant"]
            self.assertEqual(total, len(graph))

        for node in graph.nodes():
            role = graph.nodes[node]["agent_role"]
            final_state = graph.nodes[node]["final_state"]
            if role == FACT_CHECKER_ROLE:
                self.assertNotEqual(final_state, "Believer")
            elif role == RESISTANT_ROLE:
                self.assertEqual(final_state, RESISTANT)
            else:
                self.assertEqual(role, ORDINARY_ROLE)

    def test_homogeneous_attributes_equal_baseline_probabilities(self):
        cfg = ModelConfig(
            n=60,
            mean_degree=6,
            model_mode="homogeneous",
            misinformation_prob=0.15,
            correction_prob=0.35,
            seed=12,
        )
        _, metrics, graph = run_simulation(cfg)
        for node in graph.nodes():
            if graph.nodes[node]["agent_role"] == ORDINARY_ROLE:
                self.assertEqual(graph.nodes[node]["misinformation_susceptibility"], 0.15)
                self.assertEqual(graph.nodes[node]["correction_receptiveness"], 0.35)
        self.assertEqual(metrics["model_mode"], "homogeneous")
        self.assertAlmostEqual(metrics["realised_susceptibility_std"], 0.0)
        self.assertAlmostEqual(metrics["realised_correction_receptiveness_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
