# Report Draft — CITS4403

## Title
**The Impact of Network Structure and Fact-Checking Strategies on Misinformation Spread**

## 1. Background and motivation
Online misinformation is a complex-system problem because population-level outcomes emerge from many local interactions between users. The same misinformation rule may produce different outcomes depending on who is connected to whom. This project investigates whether network topology changes the scale and speed of misinformation diffusion and whether limited fact-checking resources are more effective when allocated to highly connected users rather than randomly selected users.

The project compares three stylised social-network structures: random, small-world and scale-free networks. These structures differ in path length, clustering and degree heterogeneity, allowing us to test whether network organisation interacts with intervention design.

**Research question:** How do different social network structures and fact-checking strategies affect the scale and speed of misinformation spread?

**Focused hypothesis:** With the same number of fact-checkers, degree-targeted fact-checking will reduce misinformation reach more than random placement, especially in scale-free networks where a small number of nodes have very high degree.

> Add 3–6 academic references here. Do not invent references; use papers you have actually read.

## 2. Description of the model
We implement a synchronous agent-based model on an undirected graph. Each node represents one social-media user. A node is in one of four states: Unaware, Believer, Corrected or Resistant.

### 2.1 State-transition assumptions
- Unaware users can become Believers after exposure to neighbouring Believers.
- Believers expose each Unaware neighbour once per time step.
- If an Unaware user has `k` misinformation exposures in the same time step, adoption probability is:
  `1 - (1 - beta)^k`, where `beta` is the per-exposure misinformation probability.
- Fixed fact-checkers start in the Corrected state and never become Believers.
- Corrected users may also transmit corrections.
- If a Believer receives `k` correction exposures, correction probability is:
  `1 - (1 - gamma)^k`, where `gamma` is the per-exposure correction probability.
- Resistant users never become Believers.
- State changes are applied synchronously at the end of each time step.

These rules deliberately simplify real behaviour so that the influence of network structure and targeting strategy can be isolated.

### 2.2 Network structures
We compare:
1. Erdős-Rényi random networks;
2. Watts-Strogatz small-world networks;
3. Barabási-Albert scale-free networks.

The target population size is 500 users with mean degree approximately 8. Keeping size and connectivity broadly comparable helps make topology the main structural difference.

### 2.3 Intervention strategies
We compare:
- **No intervention:** no fixed fact-checkers.
- **Random:** a fixed fraction of nodes is sampled uniformly at random.
- **Centrality:** the same fraction is assigned to the highest-degree nodes.

The key comparison therefore controls the intervention budget while changing only where fact-checkers are placed.

## 3. Modelling process and experimental design
The main experiment is a 3 × 3 factorial design:
- network = random, small-world, scale-free;
- intervention = none, random, centrality.

Suggested baseline parameters:
- `N = 500`
- target mean degree = `8`
- misinformation probability `beta = 0.15`
- correction probability `gamma = 0.35`
- resistant fraction = `0.05`
- fact-checker fraction = `0.05`
- initial believers = `5`
- maximum duration = `150` time steps.

Each condition should be repeated for at least 30 random seeds. Report the mean and standard deviation rather than relying on one stochastic run.

Primary outcomes:
- final reach;
- peak believer fraction;
- time to peak;
- extinction time.

### Independent extension
To go beyond a basic model replication, run a fact-checker-budget sensitivity analysis with 1%, 2.5%, 5%, 7.5% and 10% fact-checkers. Compare random and degree-targeted placement, especially on scale-free networks. This directly tests when targeting provides the greatest marginal benefit.

## 4. Results
**Do not fill this section before running the experiments.**

Recommended figures:
- Figure 1: mean final reach by network and intervention, with uncertainty/error bars;
- Figure 2: peak believer fraction by network and intervention;
- Figure 3: time-series plot for representative runs;
- Figure 4: fact-checker budget versus final reach for random vs centrality targeting;
- optional heat map: misinformation probability × fact-checker fraction.

For every figure, state the number of repetitions and whether bars show standard deviation or confidence intervals.

Suggested quantitative paragraph template:

> Across 30 runs per condition, [strategy] produced the lowest mean final reach on [network]. Compared with random placement, targeted placement changed final reach from [X ± SD] to [Y ± SD]. The effect was [larger/smaller] on scale-free networks than on random or small-world networks. This [supports/does not support] our hypothesis that high-degree nodes are disproportionately important intervention targets.

## 5. Conclusions, interpretation and limitations
The conclusions must be based on the observed experiment outputs. Discuss whether topology altered spread, whether targeting helped, and under what parameter ranges the effect was strongest.

Important limitations:
- all users follow the same behavioural probabilities;
- edges are static and undirected;
- no repeated sharing fatigue, forgetting, or belief relapse;
- no content-specific differences;
- degree is only one definition of influence;
- parameters are stylised rather than fitted to a real social-media dataset.

Possible extensions:
- heterogeneous susceptibility;
- directed/follower networks;
- dynamic networks;
- betweenness/PageRank targeting;
- delayed fact-checking;
- imperfect trust in corrections;
- empirical calibration using real diffusion cascades.
