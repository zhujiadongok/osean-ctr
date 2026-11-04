# Ontology-aware supervised GNN (CTR)

Task-aligned **relational GNN** for registered design fields, with a fair
**+VO / no-VO** edge ablation on the same Study–feature–VO input graph used by
CTR (targets never appear as edges).

## Graph

Built by `ctr.graph.build_input_graph`:

| Edge | Always? | Role |
|------|---------|------|
| Study → Material (`typed_as`), platform, disease, phase, … | yes | input features as nodes |
| VO `subClassOf` / `has_role` | **ablation** | ontology message passing |
| Study–Study `same_context` | yes (default) | shared disease+platform+phase |

## Model

`OntologyRGCN` in `ctr/ontology_gnn.py`:

- Learnable node embeddings + relation-specific weights (CompGCN-style)
- 3-layer message passing
- Four classification heads (allocation, masking, intervention_model, enrollment_bucket)
- Light link-prediction aux loss
- **Inductive** test studies: vector = weighted mean of linked VO/box embeddings after training encode

## Run (local)

```bash
cd osean_kg

# smoke (~few minutes on CPU)
python -u -m ctr.ontology_gnn --quick

# full chronological Setting A + baselines
python -u -m ctr.ontology_gnn

# + LOMO over paper-eligible Materials (slow)
python -u -m ctr.ontology_gnn --lomo

# debug LOMO on 3 classes
python -u -m ctr.ontology_gnn --lomo --lomo-max 3 --epochs 80

# use cached cohort CSV (no SEA-CDM profile rebuild)
python -u -m ctr.ontology_gnn --from-cache
```

Artifacts: `ckpts/sea_ml_v14/ctr/optimize/ontology_gnn/`

- `ontology_gnn_results.json`
- `report.txt`

## Run (UMich Great Lakes)

Yes — use the GPU partition. Script: `greatlakes/train_ontology_gnn.sbatch`.

**1. Sync to the cluster** (from your laptop, adjust paths/account):

```bash
# code + cached CTR table + results dir
rsync -avP osean_kg/ uniqname@greatlakes.arc-ts.umich.edu:~/osean_kg/ \
  --exclude __pycache__ --exclude '*.npy' --exclude ckpts/CompGCN_* --exclude ckpts/TransE_*

# VO owl (needed for subClassOf / has_role edges)
rsync -avP "SEA-CDM AACT vaccine remapped/research_factors/vo_mapping/FOR_ADVISOR/vo.owl" \
  uniqname@greatlakes.arc-ts.umich.edu:~/SEA-CDM\ AACT\ vaccine\ remapped/research_factors/vo_mapping/FOR_ADVISOR/
```

Keep the same relative layout: `~/osean_kg` next to `~/SEA-CDM AACT vaccine remapped/.../vo.owl`.

**2. Submit**

```bash
ssh uniqname@greatlakes.arc-ts.umich.edu
cd ~/osean_kg
mkdir -p logs
# replace YOUR_ACCOUNT with your Slurm account (e.g. lab account)
sbatch --account=YOUR_ACCOUNT greatlakes/train_ontology_gnn.sbatch

# chrono + LOMO
sbatch --account=YOUR_ACCOUNT \
  --export=ALL,OSEAN_GNN_EXTRA='--lomo --epochs 200' \
  greatlakes/train_ontology_gnn.sbatch
```

**3. Pull results back**

```bash
rsync -avP uniqname@greatlakes.arc-ts.umich.edu:~/osean_kg/ckpts/sea_ml_v14/ctr/optimize/ontology_gnn/ \
  osean_kg/ckpts/sea_ml_v14/ctr/optimize/ontology_gnn/
```

## How to read results

- Compare `ontology_rgcn_+vo` vs `ontology_rgcn_no_vo` — this is the ontology contribution.
- Also compare both to `exact_knn@20`, `ontsim_role_knn@20`, `hist_gbdt`.
- Chronological gains from VO edges are often small (ids already seen); LOMO is the harder transfer test.
