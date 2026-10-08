# Agent-level behavioural heterogeneity

## Purpose

The original model assumes that every ordinary agent has the same probability of adopting misinformation after one exposure (`misinformation_prob`) and the same probability of accepting a correction after one exposure (`correction_prob`). The heterogeneous extension tests whether conclusions about fact-checker selection remain robust when ordinary users differ in these behavioural responses.

## Distribution choice

The first implementation uses **Beta distributions** because Beta random variables are naturally bounded to the interval `[0, 1]`, which makes them suitable for probabilities.

For an ordinary agent `i`:

- misinformation susceptibility: `s_i ~ Beta(mu_s * k_s, (1 - mu_s) * k_s)`;
- correction receptiveness: `c_i ~ Beta(mu_c * k_c, (1 - mu_c) * k_c)`.

The baseline means are:

- `mu_s = misinformation_prob = 0.15`;
- `mu_c = correction_prob = 0.35`.

The default concentration parameters are:

- `k_s = susceptibility_concentration = 20`;
- `k_c = correction_concentration = 20`.

With this parameterisation, the heterogeneous population has the same intended mean response as the homogeneous baseline while allowing persistent individual differences. A larger concentration produces less between-agent variation; a smaller concentration produces more variation. The settings are configurable and are exported with each experiment row.

If a configured mean is exactly `0` or `1`, sampling is replaced with that boundary value so the probability remains valid.

## Timing and reproducibility

Agent attributes are sampled **once during initialisation** and remain fixed for the full simulation. A dedicated NumPy random-number stream derived from the simulation seed is used for these attributes. This avoids changing the original Python random-number sequence that selects fact-checkers, resistant users and initial believers, allowing the homogeneous mode to preserve the previous baseline trajectory.

## Behavioural rules

The existing multiple-exposure mechanism is retained. If an ordinary receiving agent with susceptibility `s_i` gets `k` misinformation exposures in one step, the adoption probability is:

`1 - (1 - s_i)^k`

If a Believer with correction receptiveness `c_i` gets `k` correction exposures, the correction probability is:

`1 - (1 - c_i)^k`

The receiving agent's own probability is therefore used for every exposure calculation.

## Protected agents

Fact-checkers retain their original behaviour: they start Corrected, spread correction, and cannot become Believers. Resistant agents also retain their original behaviour and do not adopt misinformation. Their stored probability attributes are metadata only and are not used to change those protected behaviours.

## Modes

- `homogeneous`: all ordinary agents use the original global probabilities exactly.
- `heterogeneous`: ordinary agents receive fixed Beta-distributed susceptibility and correction-receptiveness values.

The default remains `homogeneous` for backwards compatibility.

## Limitations

This extension represents stable individual differences only. It does **not** model learning, fatigue, changing susceptibility, repeated belief after correction, delayed intervention, correlations between susceptibility and receptiveness, demographic structure, or dynamic social networks. The Beta distribution and concentration values are modelling assumptions rather than empirically calibrated estimates.
