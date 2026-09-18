# 510K statistical report

## 1 Capability profile (win rate, mean ± 95% CI)

| mode | MLP (5) | recurrent e05 (2) | IPPO (3) |
|---|---|---|---|
| single | 0.516 ± 0.029 | (see §6.4) | 0.300 ± 0.114 |
| static | 0.774 ± 0.031 | (see §6.4) | 0.737 ± 0.094 |
| dynamic | 0.862 ± 0.016 | (see §6.4) | 0.810 ± 0.025 |
| obvious | 0.822 ± 0.006 | (see §6.4) | 0.807 ± 0.080 |

## 2 MLP vs IPPO (Welch t-test)

- single: MLP 0.516 vs IPPO 0.300 (diff +0.216), t=7.61, p=0.0074
- static: MLP 0.774 vs IPPO 0.737 (diff +0.037), t=1.52, p=0.2235
- dynamic: MLP 0.862 vs IPPO 0.810 (diff +0.052), t=6.34, p=0.0011
- obvious: MLP 0.822 vs IPPO 0.807 (diff +0.015), t=0.82, p=0.4960

## 3 DYNAMIC vs OBVIOUS (MLP, paired by seed)

- MLP: DYNAMIC 0.862 vs OBVIOUS 0.822 (diff +0.040), paired t=8.94, p=0.0009

## 4 Information-reveal curve (MLP)

- per-seed slope (win ~ p): mean -0.0296, one-sample t=-6.12, p=0.0036 (n=5)
- paired p=0 vs p=1: 0.862 vs 0.822 (diff +0.040), t=8.94, p=0.0009

## 5 Rule- vs random-bot evaluation (paired by seed)

- mlp single: rule 0.516 vs random 0.764, paired t=-42.53, p=0.0000 (n=5)
- mlp static: rule 0.774 vs random 0.950, paired t=-20.19, p=0.0000 (n=5)
- mlp dynamic: rule 0.862 vs random 0.948, paired t=-6.67, p=0.0026 (n=5)
- mlp obvious: rule 0.822 vs random 0.958, paired t=-55.52, p=0.0000 (n=5)
- ippo single: rule 0.300 vs random 0.450, paired t=-15.00, p=0.0044 (n=3)
- ippo static: rule 0.737 vs random 0.867, paired t=-22.52, p=0.0020 (n=3)
- ippo dynamic: rule 0.810 vs random 0.907, paired t=-5.21, p=0.0349 (n=3)
- ippo obvious: rule 0.807 vs random 0.950, paired t=-7.07, p=0.0194 (n=3)

## 6 Cooperation asymmetry (deferral)

### mlp
- static: asym mean +0.379 (one-sample t=41.99, p=0.0000, n=5); per-seed z: z=36.0,p=2.9e-284, z=36.1,p=3.2e-285, z=31.8,p=1.7e-222, z=35.9,p=6.9e-283, z=35.2,p=5.8e-272
- dynamic: asym mean +0.003 (one-sample t=1.51, p=0.2067, n=5); per-seed z: z=0.2,p=8.2e-01, z=0.7,p=4.7e-01, z=0.8,p=4.4e-01, z=0.6,p=5.7e-01, z=-0.5,p=6.1e-01
- obvious: asym mean -0.037 (one-sample t=-31.52, p=0.0000, n=5); per-seed z: z=-3.9,p=1.0e-04, z=-4.6,p=4.3e-06, z=-4.6,p=4.3e-06, z=-4.6,p=4.3e-06, z=-4.6,p=4.3e-06

### lstm_e05
- static: asym mean +0.375 (one-sample t=46.41, p=0.0137, n=2); per-seed z: z=24.9,p=2.7e-137, z=25.9,p=2.5e-148
- dynamic: asym mean -0.043 (one-sample t=-inf, p=0.0000, n=2); per-seed z: z=-3.8,p=1.3e-04, z=-3.8,p=1.3e-04
- obvious: asym mean -0.066 (one-sample t=-15.69, p=0.0405, n=2); per-seed z: z=-6.3,p=2.8e-10, z=-5.5,p=2.9e-08

### lstm_small
- static: asym mean +0.383 (one-sample t=inf, p=0.0000, n=2); per-seed z: z=25.9,p=2.5e-148, z=25.9,p=2.5e-148
- dynamic: asym mean -0.043 (one-sample t=-inf, p=0.0000, n=2); per-seed z: z=-3.8,p=1.3e-04, z=-3.8,p=1.3e-04
- obvious: asym mean -0.071 (one-sample t=-inf, p=0.0000, n=2); per-seed z: z=-6.3,p=2.8e-10, z=-6.3,p=2.8e-10

