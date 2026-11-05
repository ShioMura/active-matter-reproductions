## Bounded continuation: reference assessment

### System and measurement interval

- Periodic square lattice: L = 20, N = 200.
- Temperature: T = 0.4; interaction strength: epsilon = 1.
- Sampler: nonlocal Metropolis exchange at fixed particle number.
- Nine trajectories: three random, three vertical-stripe, and
  three horizontal-stripe initial states.
- Each trajectory was continued to 100,000 sweeps.
- Summary estimates use recorded samples after sweep 20,000,
  through sweep 100,000, with equal weight for each trajectory.

### Provisional observable-mean reference

| Observable | Mean | Between-chain SEM | Combined block SEM |
|---|---:|---:|---:|
| U | -361.738375 | 0.048010 | 0.043971 |
| S_low | 16.794813 | 0.026793 | 0.023187 |
| abs_D | 31.228528 | 0.097942 | 0.078962 |

Between-chain SEM is the sample standard deviation of the nine
trajectory means divided by sqrt(9). Combined block SEM is
sqrt(sum(s_i^2))/9, where s_i is the largest block SEM obtained
for trajectory i using block lengths of 1,000, 2,000, and 4,000
sweeps. The latter calculation assumes independent trajectories
and adequate within-trajectory uncertainty estimates.

These are diagnostic estimates of statistical uncertainty.
Neither estimates common equilibration bias, and they are not
combined into a certified confidence interval.

### Decision

All prespecified mean screens passed for U, S_low, and abs_D:
between-chain mean range, change between the two measurement
windows, and maximum within-chain block SEM.

Retain these estimates as a provisional observable-mean reference
for this specific system and parameter set. Passing the screens
does not establish an upper bound on equilibrium bias.

Directional sampling remains limited: several trajectories spend
most of the measurement interval in one orientation, and persistent
orientation changes are infrequent. Full-configuration equilibrium
sampling is therefore not qualified by this investigation.

The bounded continuation is complete. No further extension is
included in this task. Nonlocal exchange is used here for sampling;
its trajectories are not interpreted as local Kawasaki dynamics.

### Provenance

The numerical assessment and frozen criteria are retained in
`reference_assessment.json` and `config.json` in this directory.
The trajectory segments and RNG checkpoints are retained unchanged.
