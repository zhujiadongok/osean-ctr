# OntSim combinatorial search — recommended config

Protocol: tune on chrono val (2019–2021), evaluate once on chrono test (≥2022) and pooled unseen Materials. Paper baselines unchanged.

## Winner (default field weights — preferred for paper story)

**`gated_ic_sib`** (`graph_gated_ic_sib` / `ontsim_gated_ic_sib`)

| knob | setting |
|------|---------|
| gate | if query Material id ∈ train → exact VO; else OntSim |
| subclass | IC-weighted Jaccard (Resnik-style corpus IC over VO) |
| role | **off** (siblings already add lateral signal; sep-role did not help here) |
| neighborhood | parents (subclass closure) **+ direct siblings** |
| children | off |
| weights | paper defaults (disease/phase/platform=1, purpose/material=0.5) |

| split | exact | paper union OntSim | **gated_ic_sib** |
|-------|------:|-------------------:|-----------------:|
| Chrono Setting A field-mean acc | 0.6581 | 0.6564 | **0.6581** (= exact) |
| Pooled unseen trial-mean acc | 0.6203 | 0.6350 | **0.6390** (+1.88 pp) |

Story: ontology only fills **unseen / unmatched** Materials; IC + siblings cut false “all vaccines” neighbors without hurting chronological exact matching.

## Optional: val-tuned weights (max numeric)

Same structure as above, with `w_purpose=1.0`, `w_platform=0.5` (no route):

| split | acc |
|-------|----:|
| Chrono | **0.6696** (+1.15 pp vs exact) |
| Pooled | **0.6384** (+1.82 pp) |

Fairness note: changing platform/purpose weights changes the exact baseline too if re-matched; use only with a re-weighted exact control, or keep default weights in the main ablation.

## What did / didn’t help

| idea | outcome |
|------|---------|
| Gated OntSim | Chrono ≈ exact; pooled keeps OntSim gain |
| Role as **separate** channel vs union | Fixes toy J dilution; cohort-level ≈ union |
| IC weighting | Small chrono lift when always-on; strong with siblings |
| Siblings | Best pooled lift under **gate**; without gate hurts chrono |
| Children depth=1 | Noisy; usually worse pooled |
| Val weight + small route | Route not needed; purpose↑/platform↓ can help chrono |
| Always-on OntSim (paper) | Slight chrono dip vs exact |

Artifacts: `ontsim_search.json`, `structural_val.json`, `tuned_val.json`, `finalists.json`.
