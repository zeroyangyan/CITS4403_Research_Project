# CITS4403 Research Project

## The Impact of Network Structure and Fact-Checking Strategies on Misinformation Spread

## Research question

**How do different social network structures and fact-checking strategies affect the scale and speed of misinformation spread?**

The project also considers a more focused question:

**When the number of fact-checkers is fixed, is targeting highly connected users more effective than selecting fact-checkers at random?**

## Project overview

Misinformation spreads through interactions between many users. The overall spreading pattern may depend on how those users are connected. A strategy that works well in one type of network may be less effective in another.

This project uses an agent-based model on simulated social networks. Each node represents a social-media user, while each edge represents a connection through which misinformation or corrective information can travel. We will compare misinformation outcomes across different network structures and fact-checking strategies.

The purpose is not only to simulate misinformation. The model will be used to test whether network structure changes the effectiveness of different interventions.

## Model design

Each user can be in one of four states:

- `Unaware`: the user has not adopted the misinformation.
- `Believer`: the user believes and can spread the misinformation.
- `Corrected`: the user no longer believes the misinformation and may share corrective information.
- `Resistant`: the user does not adopt the misinformation.

During each simulation step, current Believers expose connected Unaware users to misinformation. The chance of adoption increases when a user receives several exposures. Fact-checkers and eligible Corrected users expose neighbouring Believers to corrective information. All state changes are applied at the end of the step so that the update is synchronous.

Fact-checkers are fixed users who begin in the `Corrected` state and cannot become Believers. The current baseline assumes that Believers remain active until they are corrected. We will review this assumption during model validation and may compare it with a version that includes natural forgetting or loss of interest.

## Network structures

The model compares three stylised network structures:

1. **Erdos-Renyi random network**: connections are formed randomly.
2. **Watts-Strogatz small-world network**: users have many local connections and a smaller number of long-range connections.
3. **Barabasi-Albert preferential-attachment network**: a small number of users have many connections, while most users have fewer connections.

The experiments use approximately the same population size and mean degree across network types. This helps isolate the effect of network structure.

## Fact-checking strategies

Three intervention conditions are compared:

1. **No intervention**: no fixed fact-checkers are added.
2. **Random fact-checking**: fact-checkers are selected uniformly at random.
3. **Degree-targeted fact-checking**: users with the highest degree are selected as fact-checkers.

Random and targeted strategies use the same fact-checker budget within each comparison.

## Hypotheses

We expect network structure to influence both the speed and final reach of misinformation. In networks with highly connected users, misinformation may spread rapidly when one of these users becomes a Believer.

We also expect degree-targeted fact-checking to reduce misinformation more effectively than random selection, particularly in preferential-attachment networks. However, the size of this advantage may change across network types and intervention budgets.

## Experimental design

The main experiment uses a 3 x 3 factorial design:

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

## Evaluation metrics

The main outcomes are:

- final reach: the fraction of users who became Believers at least once;
- peak number and fraction of Believers;
- time required to reach the peak;
- whether misinformation remains active at the maximum simulation time;
- extinction time, where extinction occurs during the simulation.

The meaning of extinction time will be interpreted carefully because the current baseline does not include natural forgetting. Comparisons will therefore focus mainly on final reach and peak prevalence unless this assumption is extended.

## Repository structure

```text
project-root/
|-- src/
|   |-- model.py
|   |-- experiments.py
|   `-- plots.py
|-- utils/
|   `-- helpers.py
|-- data/
|   `-- README.md
|-- notebooks/
|   `-- analysis.ipynb
|-- results/
|   `-- README.md
|-- requirements.txt
`-- README.md
```

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

## Running the project

Run the following commands from the project root.

### Quick test

```bash
cd src
python experiments.py --experiment main --n 100 --repeats 3 --output ../results/quick_test.csv
```

### Main experiment

```bash
cd src
python experiments.py --experiment main --n 500 --repeats 30 --output ../results/main_experiment.csv
```

### Fact-checker budget experiment

```bash
cd src
python experiments.py --experiment budget --n 500 --repeats 30 --output ../results/budget_sweep.csv
```

### Generate summary figures

After running the main experiment:

```bash
cd src
python plots.py
```

The generated CSV files and figures are stored in the `results` directory.

## Current progress

- The research question and focused sub-question have been defined.
- The shared GitHub repository and initial directory structure have been created.
- The three network generators and four user states have been implemented.
- The baseline misinformation and correction rules have been implemented.
- The main factorial experiment and budget-sensitivity experiment have been prepared.
- An initial notebook has been created for model exploration and analysis.

The next stage is to validate the model assumptions, inspect the three generated network structures, run small tests and confirm that the selected parameter ranges produce meaningful behaviour before completing the full experiments.

## Collaboration

Both team members will make regular, meaningful commits. GitHub Issues and branches will be used to divide tasks, review changes and record the development process.

