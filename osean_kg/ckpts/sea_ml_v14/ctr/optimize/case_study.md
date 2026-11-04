# OSEAN-CTDR case study (held-out)

**Query NCT (hidden design):** `NCT05584202`
- Disease: covid
- Material: COVID-19 RNA vaccine (`VO_0005198`)
- Platform: RNA vaccine (`VO_0000186`)
- Phase: PHASE2
- Purpose: PREVENTION
- Method: `hybrid_compgcn_128_a0.75_k10`

## Recommendation vs registered design

| variable | recommended | actual_registered | match |
| --- | --- | --- | --- |
| allocation | RANDOMIZED | RANDOMIZED | True |
| masking | QUADRUPLE | QUADRUPLE | True |
| intervention_model | PARALLEL | SEQUENTIAL | False |
| enrollment_bucket | ge_500 | 50_149 | False |

**Exact matches:** 2 / 4

## Supporting historical trials (evidence)

| nct_id | score |
| --- | --- |
| NCT04748471 | 4.0 |
| NCT05324319 | 4.0 |
| NCT04405076 | 4.0 |
| NCT05137236 | 4.0 |
| NCT04649151 | 3.0 |
| NCT04813796 | 3.0 |
| NCT05132855 | 3.0 |
| NCT04761822 | 3.0 |
| NCT04785144 | 3.0 |
| NCT04470427 | 3.0 |

## Note

This demonstrates explainable recovery of historical design practice, 
not clinical optimality of the recommendation.
