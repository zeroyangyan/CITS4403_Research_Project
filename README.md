# CITS4403 Research Project
## The Impact of Network Structure and Fact-Checking Strategies on Misinformation Spread

### Research question
**How do different social network structures and fact-checking strategies affect the scale and speed of misinformation spread?**

Focused sub-question:

**When the number of fact-checkers is fixed, is targeting highly connected users more effective than selecting fact-checkers at random?**

## Model
This project uses an agent-based model on a social network. Each node is a social-media user and can be:

- `Unaware`: has not adopted the misinformation.
- `Believer`: believes and can spread the misinformation.
- `Corrected`: no longer believes; can optionally spread corrections.
- `Resistant`: does not adopt the misinformation.

Fact-checkers are fixed agents that begin in the `Corrected` state.

### Network structures
1. Erdős-Rényi random network
2. Watts-Strogatz small-world network
3. Barabási-Albert scale-free network

### Interventions
1. No intervention
2. Random fact-checkers
3. Degree-targeted fact-checkers

## Main metrics
- final reach: fraction of users who ever became believers
- peak number / fraction of believers
- time to peak
- extinction time
- whether misinformation is still active at the maximum simulation time

## Repository structure
```text
project-root/
├── src/
│   ├── model.py
│   ├── experiments.py
│   └── plots.py
├── utils/
│   └── helpers.py
├── data/
├── notebooks/
│   └── analysis.ipynb
├── results/
├── requirements.txt
├── REPORT_DRAFT.md
└── README.md
```

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the main experiment
From the project root:

```bash
cd src
python experiments.py --experiment main --n 500 --repeats 30 --output ../results/main_experiment.csv
python plots.py
```

For a quick test:

```bash
python experiments.py --experiment main --n 100 --repeats 3 --output ../results/quick_test.csv
```

## Independent extension
A fact-checker-budget sweep is included:

```bash
python experiments.py --experiment budget --n 500 --repeats 30 --output ../results/budget_sweep.csv
```

This tests whether targeted fact-checking has a larger advantage when intervention resources are scarce.

## Reproducibility
Every run records the random seed and all core parameters in the output CSV. Use the same seed and configuration to reproduce a run.

## Suggested GitHub workflow
Both members should make regular, meaningful commits. Use Issues and Pull Requests rather than committing everything directly to `main`.

Example branches:
- `feature/network-generators`
- `feature/agent-states`
- `feature/fact-checking`
- `experiment/main-factorial`
- `analysis/plots`

Example issues:
- Implement synchronous misinformation spread
- Validate equal/near-equal mean degree across network types
- Add centrality targeting
- Add 30-seed experiment
- Add budget sensitivity experiment
- Produce report figures
