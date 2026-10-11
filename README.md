# CITS4403 Research Project

## The Impact of Network Structure and Fact-Checking Strategies on Misinformation Spread

This project investigates misinformation diffusion using a stochastic **agent-based model (ABM)** on synthetic social networks. It evaluates how network structure, fact-checker placement, individual behavioural differences, and experimental parameters influence the **reach and speed** of misinformation.

**Final experimental record:** **31,500 completed simulation runs**, with **100 repetitions per condition**. The authoritative results are in [`results/full_1000/`](results/full_1000/); see [`results/README.md`](results/README.md) for the complete data dictionary, experiment breakdown, and figures. Earlier CSVs in `results/` are not substitutes for this final dataset.

## Research questions

1. How do different social-network structures and fact-checking strategies affect misinformation spread?
2. Given the same fact-checker budget, does degree-targeted placement outperform random placement?


## Model overview

Each node represents an agent and each edge a potential interaction. Agents occupy one of four states:

| State | Meaning |
| --- | --- |
| `Unaware` | Has not adopted the misinformation |
| `Believer` | Currently believes and may spread misinformation |
| `Corrected` | No longer believes and may transmit corrective information |
| `Resistant` | Does not adopt misinformation |

Believers expose connected Unaware agents to misinformation; fact-checkers and eligible Corrected agents can deliver corrective exposures to Believers. Changes are applied **synchronously** at the end of each simulation step. With `k` independent exposures and per-exposure response probability `p`, the probability of response is `1 - (1 - p)^k`.

Fixed fact-checkers begin Corrected and are protected against misinformation adoption. The model uses static, undirected synthetic networks; spontaneous forgetting, re-adoption after correction, and dynamic rewiring are not part of the documented final-run assumptions.

### Networks

- **Erdős–Rényi (`random`)**: edges are sampled randomly.
- **Watts–Strogatz (`small_world`)**: locally clustered links with rewired shortcuts.
- **Barabási–Albert (`scale_free`)**: preferential attachment produces highly connected hubs.

### Interventions

- **`none`**: no fixed fact-checkers (baseline).
- **`random`**: fact-checkers are selected at random.
- **`centrality`**: fact-checkers are selected using degree-based targeting (high-degree nodes).

### Behavioural variation

Experiments compare `homogeneous`, `low`, and `high` heterogeneity conditions. These are modelled scenarios, **not empirically estimated properties of social-media users**. See [`docs/HETEROGENEITY.md`](docs/HETEROGENEITY.md) for implementation-specific notes. For the actual final run, the archived scripts in [`results/full_1000/source_snapshot/`](results/full_1000/source_snapshot/) and its `metadata.json` are the provenance record; do not assume an independently edited root-level `src/` file is byte-for-byte identical.

## Final experimental design

The completed suite is identified as `full` in [`results/full_1000/metadata.json`](results/full_1000/metadata.json). It uses **100 repetitions per recorded configuration**, baseline **1,000 agents**, **300 maximum steps**, and an initial-believer fraction of **1%**. The baseline parameters include target mean degree **8**, misinformation adoption probability **0.15**, correction probability **0.35**, resistant fraction **5%**, and intervention fraction **5%**.

| Experiment family | What it tests | Runs |
| --- | --- | ---: |
| `main` | Network × strategy × heterogeneity baseline comparisons | 2,700 |
| `budget` | Fact-checker fractions of 1%, 2.5%, 7.5%, and 10% | 7,200 |
| `sensitivity` | Adoption probabilities 0.10 / 0.20 and correction probabilities 0.20 / 0.50 | 10,800 |
| `structure` | Agent populations of 500 / 5,000 and mean degrees of 4 / 12 against baseline settings | 10,800 |
| **Total** | **13 recorded scenarios, 31,500 complete runs** | **31,500** |

The `main` family contains `3 networks × 3 strategies × 3 heterogeneity settings × 100 repeats = 2,700 runs`. Budget comparisons use the two active intervention strategies; the other scenario families also include the no-intervention condition. Results are synthetic simulation evidence, not measurements from a real social network.

## Outcome measures

The final dataset records, among other fields:

- **Final reach (`final_reach`)**: fraction of agents who ever adopted misinformation; this is cumulative, even if subsequently corrected.
- **Peak prevalence (`peak_believer_fraction`, `peak_believers`)**: maximum simultaneous share / count of active Believers.
- **Time to peak (`time_to_peak`)**: simulated step at which the recorded peak occurred.
- **Persistence (`ended_with_active_believers`)**: whether Believers remain at the end of the run.
- **Extinction time (`extinction_time`)**: where misinformation dies out during the simulated period.

Reported means, uncertainty intervals, and centrality-minus-random contrasts are supplied with the final dataset; consult those files rather than inferring statistical significance from plots alone.

## Final results and figures

Use the following files for the **final report and presentation**:

| Location | Purpose |
| --- | --- |
| [`results/full_1000/raw_results.csv`](results/full_1000/raw_results.csv) | 31,500 run-level records, the primary analysis dataset |
| [`results/full_1000/metadata.json`](results/full_1000/metadata.json) | Run configuration, completion counts, environment, and provenance |
| [`results/full_1000/figures/summary.csv`](results/full_1000/figures/summary.csv) | Grouped statistics and 95% confidence intervals |
| [`results/full_1000/figures/strategy_contrasts.csv`](results/full_1000/figures/strategy_contrasts.csv) | Comparisons of centrality-targeted versus random allocation |
| [`results/full_1000/figures/`](results/full_1000/figures/) | Main, budget, sensitivity, structure, and contrast plots |
| [`results/full_1000/representative_histories.csv`](results/full_1000/representative_histories.csv) | Selected stepwise trajectories; not histories of every run |
| [`results/full_1000/source_snapshot/`](results/full_1000/source_snapshot/) | Source files archived with the completed run |

The older [`results/combined_final.csv`](results/combined_final.csv) corresponds to an earlier experimental stage and **must not be presented as the 31,500-run result**. The final figures are those inside `full_1000/figures/`, not necessarily the older top-level `results/figures/` or `results/combined_figures/` folders.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/                    # Model, experiment, and plotting code
├── tests/                  # Validation tests
├── docs/                   # Modelling and heterogeneity notes
├── data/                   # No external empirical dataset required
├── results/
│   ├── README.md
│   ├── full_1000/          # Authoritative final 31,500-run suite
│   │   ├── raw_results.csv
│   │   ├── metadata.json
│   │   ├── representative_histories.csv
│   │   ├── figures/
│   │   └── source_snapshot/
│   └── ...                 # Historical / intermediate outputs
└── notebooks/              # Optional exploratory work
```

## Installation and reproduction

Python 3.10+ is a practical starting point for local development. The **recorded final execution environment** was Python **3.13.14 on macOS arm64**, with package versions and file hashes documented in `results/full_1000/metadata.json`. For strict reproduction, check those recorded versions and the archived source scripts rather than assuming the current root-level source matches them.

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The command recorded for the completed final suite is:

```bash
python experiments.py --suite full --repeats 100 --workers 4 --output ../results/full_1000
```

This was run from the experiment script's source directory. The equivalent command *from the project root*, **when using a matching, verified implementation**, is:

```bash
python src/experiments.py --suite full --repeats 100 --workers 4 --output results/full_1000_reproduction
```

**Important reproducibility qualification:** The previously inspected root-level source snapshot had interface/merge inconsistencies. The completed 31,500-run output is verified by its data and metadata, but rerunning with the current `src/` code should **not** be assumed to work until its syntax, CLI, and tests are checked against `full_1000/source_snapshot/`. Always choose a **new output directory** and never overwrite `results/full_1000/`.

To validate the archived results without rerunning simulations:

```bash
python -c "import csv,json,pathlib; p=pathlib.Path('results/full_1000'); m=json.loads((p/'metadata.json').read_text()); n=sum(1 for _ in csv.DictReader((p/'raw_results.csv').open(newline=''))); print(m['status'], m['completed_runs'], n); assert m['status']=='completed' and n==m['completed_runs']==31500"
```

Unit tests for the working source tree (after resolving any source inconsistencies):

```bash
python -m unittest discover -s tests -v
```

## Scope and limitations

The networks, behaviour distributions, exposure probabilities, and intervention mechanisms are stylised modelling assumptions, not calibrated real-world estimates. The final run compares strategies under defined simulated conditions; it does not show that degree targeting will always outperform other approaches in real platforms. Network connections are static and undirected, and mechanisms such as temporal network changes, fatigue, and belief relapse are not modelled. Simulation conclusions should be interpreted alongside the study's sensitivity analyses and confidence intervals.

## Documentation

- [`results/README.md`](results/README.md) — **authoritative guide to final datasets, 31,500-run suite, and figures**.
- [`docs/HETEROGENEITY.md`](docs/HETEROGENEITY.md) — behavioural heterogeneity assumptions and implementation caveats.

This repository was developed as a CITS4403 research project. The final research report should be used for the complete methodological discussion, interpretation of findings, and academic references.
