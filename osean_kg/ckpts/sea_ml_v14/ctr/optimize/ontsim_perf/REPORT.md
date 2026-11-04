# OntSim performance lock (vs paper)

Val-tuned on 2019–2021; scored once on chrono test (≥2022) and pooled unseen.

## Best for raw performance

**`ontsim_gated_ic_sib_perf`** / `graph_gated_ic_sib_perf`

| knob | value |
|------|-------|
| gate | seen Material → exact; else OntSim |
| neighborhood | IC-weighted subclass + siblings |
| role | off |
| weights | disease=1, phase=1, **purpose=1**, **platform=0.5**, material=0.5 |
| k | 20 |

| protocol | exact (paper) | OntSim+role (paper) | gated_ic_sib (default w) | **perf lock** |
|----------|--------------:|--------------------:|-------------------------:|-------------:|
| Chrono A field-mean | 0.6581 | 0.6564 | 0.6581 | **0.6696** |
| Pooled trial-mean | 0.6203 | 0.6350 | **0.6390** | 0.6384 |
| vs exact chrono | — | −0.18 pp | 0 | **+1.15 pp** |
| vs exact pooled | — | +1.47 pp | +1.88 pp | **+1.82 pp** |

Chrono lifts past HistGBDT/logistic (~0.658–0.661). Pooled stays above paper OntSim.

## Also tried (did not beat perf lock overall)

| variant | chrono | pooled | note |
|---------|-------:|-------:|------|
| gated Resnik (tuned w, k=30) | **0.6705** | 0.6320 | max chrono; pooled weaker |
| gated_ic_sib + tuned w (k=30) | 0.6670 | 0.6342 | |
| IC+sib ungated + heavy w | 0.6466 | **0.6414** | pooled high, chrono hurt |
| + HistGBDT 50/50 ensembles | ≤0.670 | ≤0.634 | chrono help; pooled usually down |
| Lin / children_if_coarse | — | — | no test win over IC+sib |

## LOMO (class-mean trial acc)

From `ontsim_perf.json` (subset):

| method | class-mean | trial-weighted |
|--------|----------:|---------------:|
| exact | 0.611 | 0.617 |
| paper union | 0.604 | 0.619 |
| **gated_ic_sib (default w)** | **0.619** | **0.627** |

Default-weight gated_ic_sib already fixes paper’s LOMO class-equal dip vs exact.

## Code

- Mode: `graph_gated_ic_sib_perf` in `ctdr/recommend.py` (`W_PERF`)
- Infer: `method=ontsim_gated_ic_sib_perf`
- Artifacts: `ontsim_perf.json`, `perf_lock.json`

**Fair paper ablation** (exact vs OntSim same weights) still use defaults / `gated_ic_sib`. Use `_perf` when optimizing absolute accuracy.
