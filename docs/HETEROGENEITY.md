# Agent-level behavioural heterogeneity

## Which implementation produced the archived results?

**The 810 rows in `results/combined_final.csv` use the `heterogeneity` field with values `homogeneous`, `low`, and `high`.** The CSV explicitly identifies its distribution as `paired_symmetric_uniform`, with a stored `heterogeneity_strength` of 0.0, 0.25 and 0.90 respectively. They **do not** represent the Beta-distribution approach described in the previous version of this document.

The repository also contains a separate, partially merged `model_mode="homogeneous"` / `model_mode="heterogeneous"` interface with Beta-distributed parameters. These two interfaces must not be conflated, and the uploaded code currently does not parse successfully. Do not claim that Beta sampling was used in `combined_final.csv`.

## Archived paired-symmetric method

For each probability mean `mu`, a heterogeneity level specifies `strength`: `homogeneous = 0`, `low = 0.25`, and `high = 0.90`. The implementation computes

```text
width = strength * min(mu, 1 - mu)
d ~ Uniform(0, width)
paired probabilities = (mu - d, mu + d)
```

The pairs are shuffled among agents; when the population size is odd, one additional value equal to `mu` is used. This ensures values stay within `[0, 1]` and that the average over all assigned values equals the configured mean (subject to floating-point precision). Misinformation adoption uses `mu = 0.15`, while correction receptiveness uses `mu = 0.35` in the archived experiment.

Attributes remain fixed within each simulated run. For `k` exposures to a receiving agent with response probability `p_i`, the probability of a transition is `1 - (1 - p_i)**k`. Fact-checkers and resistant agents still obey their protected state rules.

The archived experiment records `graph_sha256` and `attribute_sha256` for comparison checks. It shares graph realisations across compared conditions and attributes across strategies within a heterogeneity level; it **does not guarantee identical initially selected Believer or Resistant nodes** across strategies.

## Separate Beta-mode development

Another interface in the uploaded `src/model.py` specifies Beta attributes via:

```text
susceptibility ~ Beta(mu_s * k_s, (1 - mu_s) * k_s)
receptiveness ~ Beta(mu_c * k_c, (1 - mu_c) * k_c)
```

Default intended means are `mu_s = 0.15` and `mu_c = 0.35`, with concentration parameters `k_s = k_c = 20`. This mode is described by `model_mode` and `heterogeneity_distribution`, not the `low`/`high` `heterogeneity` settings above. The Beta approach can be useful for a *separate* sensitivity analysis after the merged implementation and tests are repaired. There is no basis in the supplied CSV to label its 810 runs as Beta-sampled.

## Limitations and status

Neither distribution represents fitted social-media behaviour. The uniform-paired scheme controls population means exactly but constrains variability and symmetry; the Beta scheme would generate random sample means and different tails. These choices may lead to different outcomes.
