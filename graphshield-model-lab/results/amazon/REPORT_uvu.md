# GraphShield static-graph study (generated from the result JSON)

11944 nodes, 2073474 directed edges, 8639 labelled nodes (9.5% fraud). Split fractions [0.4, 0.2, 0.4] (train/val/test), a fresh stratified split per seed; seeds [0, 1, 2]. Mean ± std over seeds.

| model | PR-AUC | ROC-AUC | F1 | MCC | P@100 | ECE |
|---|---|---|---|---|---|---|
| hgb | 0.905 ± 0.005 | 0.977 ± 0.001 | 0.856 ± 0.005 | 0.846 ± 0.004 | 0.973 ± 0.012 | 0.021 ± 0.001 |
| hgb+agg | 0.917 ± 0.008 | 0.981 ± 0.003 | 0.857 ± 0.008 | 0.845 ± 0.007 | 0.983 ± 0.009 | 0.023 ± 0.001 |
| rf+agg | 0.905 ± 0.002 | 0.971 ± 0.002 | 0.840 ± 0.012 | 0.827 ± 0.010 | 1.000 ± 0.000 | 0.020 ± 0.000 |
| sage | 0.862 ± 0.008 | 0.966 ± 0.001 | 0.812 ± 0.002 | 0.804 ± 0.002 | 0.973 ± 0.012 | 0.166 ± 0.006 |
| gcn | 0.574 ± 0.025 | 0.900 ± 0.006 | 0.604 ± 0.009 | 0.562 ± 0.010 | 0.717 ± 0.037 | 0.209 ± 0.007 |

## PR-AUC difference to `hgb` (paired by seed, bootstrap 95% CI of the mean over seeds)

| model | mean diff | 95% CI | seeds better |
|---|---|---|---|
| hgb+agg | +0.012 | [+0.008, +0.015] | 3/3 |
| rf+agg | -0.000 | [-0.006, +0.006] | 2/3 |
| sage | -0.043 | [-0.059, -0.028] | 0/3 |
| gcn | -0.331 | [-0.359, -0.315] | 0/3 |

## PR-AUC difference to `hgb+agg` (paired by seed, bootstrap 95% CI of the mean over seeds)

| model | mean diff | 95% CI | seeds better |
|---|---|---|---|
| hgb | -0.012 | [-0.015, -0.008] | 0/3 |
| rf+agg | -0.012 | [-0.019, -0.002] | 0/3 |
| sage | -0.054 | [-0.071, -0.036] | 0/3 |
| gcn | -0.343 | [-0.367, -0.330] | 0/3 |

