# CITS4403 Research Project

## The Impact of Network Structure and Fact-Checking Strategies on Misinformation Spread

## Research Question

**How do different social network structures and fact-checking strategies affect the scale and speed of misinformation spread?**

The project also considers a more focused question:

**When the number of fact-checkers is fixed, is targeting highly connected users more effective than selecting fact-checkers at random?**

---

## Project Overview

Misinformation spreads through interactions between many users. The overall spreading pattern may depend on how those users are connected. A strategy that works well in one type of network may be less effective in another.

This project uses an agent-based model on simulated social networks. Each node represents a social-media user, while each edge represents a connection through which misinformation or corrective information can travel. We will compare misinformation outcomes across different network structures and fact-checking strategies.

The model compares misinformation outcomes across different network structures and fact-checking strategies. The purpose is not only to simulate misinformation spread, but to investigate whether network structure changes the effectiveness of different intervention strategies.

The current implementation supports repeated stochastic simulations, quantitative evaluation metrics, CSV result export, and summary visualisations.

---

## Model Design

Each user can be in one of four states:

- `Unaware`: the user has not adopted the misinformation.
- `Believer`: the user believes and can spread the misinformation.
- `Corrected`: the user no longer believes the misinformation and may share corrective information.
- `Resistant`: the user does not adopt the misinformation.

During each simulation step, current `Believer` users expose connected `Unaware` users to misinformation. The probability of adoption increases when a user receives multiple exposures.

Fact-checkers and eligible `Corrected` users expose neighbouring `Believer` users to corrective information.

All state changes are applied at the end of each simulation step. This means that the model uses synchronous updating, so changes during a step do not immediately affect other interactions within the same step.

Fact-checkers are fixed users who begin in the `Corrected` state and cannot become `Believer` users.

The current baseline assumes that `Believer` users remain active until they are corrected. This assumption will be considered when interpreting the results and may later be extended by introducing natural forgetting or loss of interest.

---

## Network Structures

The model compares three stylised network structures:

1. **Erdos-Renyi random network:** connections are formed randomly. This represents a network where users have approximately similar opportunities to form connections and where there is no strong hub structure.

2. **Watts-Strogatz small-world network:** users have many local connections and a smaller number of long-range connections. This network can produce relatively high clustering while maintaining short paths between users.

3. **Barabasi-Albert preferential-attachment network:** a small number of users have many connections, while most users have fewer connections. This produces a scale-free style degree distribution where a small number of highly connected hub users coexist with many low-degree users.

The experiments use approximately the same population size and mean degree across the three network types. This helps reduce differences caused purely by the total number of connections and allows the analysis to focus more directly on network structure.

---

## Fact-Checking Strategies

Three intervention conditions are compared:

### No Intervention

No fixed fact-checkers are introduced. This condition provides a baseline showing how misinformation spreads without intervention.

### Random Fact-Checking

A fixed number of users are selected uniformly at random and assigned as fact-checkers.

### Degree-Targeted Fact-Checking

Users with the highest degree are selected as fact-checkers. These users are chosen because highly connected users may have greater influence over information diffusion. Random and targeted strategies use the same fact-checker budget within each comparison so that the effect of the selection strategy can be examined fairly.

---

## Hypotheses

We expect network structure to influence both the speed and final reach of misinformation. In networks with highly connected users, misinformation may spread rapidly when one of these users becomes a Believer.

We also expect degree-targeted fact-checking to reduce misinformation more effectively than random selection, particularly in preferential-attachment networks. However, the size of this advantage may change across network types and intervention budgets.

---

## Experimental Design

The main experiment uses a **3 x 3 factorial design**:

- three network structures: random, small-world and preferential attachment;
- three intervention conditions: none, random and degree-targeted.

The baseline configuration uses:

- 500 users;
- a target mean degree of approximately 8;
- 5 initial Believers;
- a misinformation probability of 0.15 per exposure;
- a correction probability of 0.35 per exposure;
- a resistant-user fraction of 0.05;
- a fact-checker fraction of 0.05 for intervention conditions;
- a maximum of 150 simulation steps.

Because the model is stochastic, each condition will be repeated using multiple random seeds. The main analysis will use at least 30 runs per condition and report both the mean and variation of the results.

An additional budget experiment compares random and degree-targeted fact-checking at fact-checker fractions of 1%, 2.5%, 5%, 7.5% and 10%. This tests whether targeted intervention is especially useful when fact-checking resources are limited.

---

## Evaluation Metrics

The main outcomes are:

- final reach: the fraction of users who became Believers at least once;
- peak number and fraction of Believers;
- time required to reach the peak;
- whether misinformation remains active at the maximum simulation time;
- extinction time, where extinction occurs during the simulation.

The meaning of extinction time will be interpreted carefully because the current baseline does not include natural forgetting. Comparisons will therefore focus mainly on final reach and peak prevalence unless this assumption is extended.

---

## Installation

Python 3.10 or later is recommended.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## Current Validation Tasks

1. Inspect the degree distributions of the three network structures.
2. Confirm that network types have approximately comparable mean degrees.
3. Test whether results are sensitive to the chosen random seed.
4. Examine whether the baseline misinformation probability produces meaningful diffusion rather than immediate extinction or universal infection.
5. Examine the effect of correction probability.
6. Check whether the resistant-user fraction changes the qualitative conclusions.

The next stage is to validate the model assumptions, inspect the three generated network structures, run small tests and confirm that the selected parameter ranges produce meaningful behaviour before completing the full experiments.

---

## Preliminary Results

Small-scale simulations have been used to verify that the model and experimental pipeline behave as expected.

The preliminary experiments show that the different intervention strategies can produce clearly different misinformation outcomes.

In particular, degree-targeted intervention appears capable of reducing misinformation reach substantially in networks containing highly connected hub users.

These results are currently treated only as preliminary model-validation results.

They are not yet interpreted as final conclusions because:

- the Checkpoint 2 demonstration uses a relatively small number of repetitions;
- stochastic variation is still being evaluated;
- parameter sensitivity has not yet been fully analysed;
- additional validation is required before the final experiments.

The final analysis will therefore use at least 30 runs per condition.

---

## Next Stage

The next stage of the project is to complete the full experimental analysis.

The main tasks are:

- run the complete experiment;
- complete the fact-checker budget experiment;
- perform parameter-sensitivity tests;
- calculate summary statistics;
- compare intervention effectiveness across network types;
- produce final figures and tables for the report;
- prepare the final project demonstration.

---

## Agent-Level Behavioural Heterogeneity

The model now supports two behavioural modes so the existing homogeneous baseline can be compared with an extended model containing persistent individual differences.

### Homogeneous mode (default)

`model_mode="homogeneous"` preserves the original assumptions. Every ordinary agent uses:

- `misinformation_prob = 0.15` as its single-exposure misinformation-adoption probability;
- `correction_prob = 0.35` as its single-exposure correction probability.

This mode is the default so existing experiments remain backwards compatible.

### Heterogeneous mode

`model_mode="heterogeneous"` assigns each ordinary agent two fixed attributes once during initialisation:

- **misinformation susceptibility**;
- **correction receptiveness**.

Both are sampled from Beta distributions bounded to `[0, 1]`. The distributions are parameterised so their means equal the homogeneous baseline probabilities. The default concentration parameter is `20` for both attributes. Larger concentration values produce less variation between agents; smaller values produce more variation.

The existing multiple-exposure rule is retained. For `k` exposures, the simulation still uses `1 - (1 - p)^k`, but `p` is now the **receiving agent's own fixed probability** in heterogeneous mode.

Fact-checkers and resistant agents preserve their previous protected behaviour. Heterogeneity does not make them susceptible to misinformation.

A dedicated random-number stream is used to sample agent attributes. This keeps agent attributes reproducible under a fixed configuration and seed without altering the baseline random sequence used for fact-checker selection, resistant-agent selection, or initial Believers.

The full distribution assumptions and limitations are documented in [`docs/HETEROGENEITY.md`](docs/HETEROGENEITY.md).

### Running the two modes

Homogeneous baseline:

```bash
python src/experiments.py --experiment main --mode homogeneous --output results/main_homogeneous.csv
```

Heterogeneous model:

```bash
python src/experiments.py --experiment main --mode heterogeneous --output results/main_heterogeneous.csv
```

A small smoke experiment covering both modes and both fact-checker strategies can be run with:

```bash
python src/experiments.py --experiment smoke --n 100 --repeats 2 --output results/heterogeneity_smoke.csv
```

Experiment CSV files now include `model_mode`, the heterogeneity distribution and concentration settings, and realised means/standard deviations of the ordinary-agent behavioural attributes.

### Validation tests

Run the regression and invariant tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- preservation of an agreed homogeneous baseline trajectory;
- reproducibility under a fixed seed;
- probability bounds;
- state-count conservation;
- protected fact-checker and resistant-agent behaviour;
- homogeneous attributes remaining equal to the original baseline probabilities.

### Current limitations of heterogeneity

The heterogeneous extension models stable individual differences only. It does not add dynamic networks, repeated belief after correction, delayed intervention, behavioural learning, time-varying susceptibility, or empirically calibrated demographic effects. The Beta distributions and their concentration parameters are modelling assumptions and should therefore be included in sensitivity analysis rather than treated as measured real-world values.
