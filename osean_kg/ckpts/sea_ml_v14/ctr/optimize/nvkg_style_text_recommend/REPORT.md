# NVKG-style text recommendation (local vaccine CTDR)

Same scoring protocol as NVKG Table 6 (max sim + rank/MRR), but vaccine CTDR cohort and embedding model ≠ their drug-disease subset + fastText; they also cluster near-duplicate texts into shared nodes.

- train=1135 test=283
- test with primary endpoints=283
- test with inclusion criteria=282
- similarity: sentence-transformers/all-MiniLM-L6-v2

## Published NVKG (HRGAT, Table 6) — reference only

| Task | Top-N | avg sim | mean rank | MRR |
|------|------:|--------:|----------:|----:|
| Primary EP | 3 | 0.721 | 1.73 | 0.697 |
| Primary EP | 5 | 0.751 | 2.62 | 0.513 |
| Primary EP | 10 | 0.779 | 4.78 | 0.344 |
| Inclusion | 10 | 0.762 | 3.42 | 0.441 |
| Inclusion | 30 | 0.812 | 11.90 | 0.174 |
| Inclusion | 50 | 0.829 | 21.40 | 0.120 |

## Our methods (same metric definitions)

### osean_exact_knn

| Task | Top-N | avg sim | mean rank | MRR | n_queries |
|------|------:|--------:|----------:|----:|---------:|
| Primary EP | 3 | 0.405 | 1.92 | 0.638 | 1113 |
| Primary EP | 5 | 0.449 | 2.97 | 0.469 | 1113 |
| Primary EP | 10 | 0.506 | 5.14 | 0.326 | 1113 |
| Inclusion | 10 | 0.455 | 5.44 | 0.306 | 2173 |
| Inclusion | 30 | 0.551 | 14.97 | 0.152 | 2173 |
| Inclusion | 50 | 0.586 | 23.98 | 0.109 | 2173 |

### osean_ontsim_role_knn

| Task | Top-N | avg sim | mean rank | MRR | n_queries |
|------|------:|--------:|----------:|----:|---------:|
| Primary EP | 3 | 0.412 | 1.88 | 0.657 | 1113 |
| Primary EP | 5 | 0.458 | 2.98 | 0.473 | 1113 |
| Primary EP | 10 | 0.506 | 5.05 | 0.341 | 1113 |
| Inclusion | 10 | 0.459 | 5.41 | 0.306 | 2173 |
| Inclusion | 30 | 0.551 | 14.47 | 0.158 | 2173 |
| Inclusion | 50 | 0.586 | 23.34 | 0.115 | 2173 |

### title_semantic_knn

| Task | Top-N | avg sim | mean rank | MRR | n_queries |
|------|------:|--------:|----------:|----:|---------:|
| Primary EP | 3 | 0.413 | 2.02 | 0.621 | 1113 |
| Primary EP | 5 | 0.470 | 3.08 | 0.459 | 1113 |
| Primary EP | 10 | 0.538 | 5.45 | 0.306 | 1113 |
| Inclusion | 10 | 0.492 | 5.08 | 0.324 | 2173 |
| Inclusion | 30 | 0.589 | 14.34 | 0.159 | 2173 |
| Inclusion | 50 | 0.624 | 22.61 | 0.122 | 2173 |

