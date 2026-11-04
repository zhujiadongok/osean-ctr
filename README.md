# OSEAN-CTR

**OSEAN-CTR** = **O**ntology-based **SEA** **N**etwork **C**linical **T**rial design **R**ecommendation.

Reproduction package for the preprint / JAMIA draft:
*OSEAN-CTR: Ontology-based SEA Network Clinical Trial Design Recommendation for SEA-CDM–based vaccine trial evidence integration*.

This repository contains analysis code (`osean_kg/ctr`), feature sidecars used by CTR, frozen result artifacts cited in the paper, and a Zenodo upload package under `zenodo/`. It does **not** include the full AACT dump (too large; local snapshot **2026-04-12**—see paper Table S8).

**Code:** https://github.com/zhujiadongok/osean-ctr  
**Supporting data:** upload the zip in `zenodo/` to Zenodo, then insert the assigned DOI in the manuscript.

## Layout

```
osean_kg/ctr/                  # CTR package (eval / recommend / regress / ontology_gnn)
osean_kg/vo_architecture.py    # VO class load + PROFILE_TO_OFFICIAL roll-up
osean_kg/ckpts/sea_ml_v14/     # disease_vo_map, AE metrics, ctr/optimize/*.json
data/sea_cdm_v14/              # study_context_profile + intervention profile sidecars
zenodo/                        # supporting-data zip + metadata for Zenodo upload
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# optional overrides
set OSEAN_CTR_ROOT=%CD%
set SEA_CDM_V14=%CD%\data\sea_cdm_v14
set PYTHONPATH=%CD%\osean_kg;%PYTHONPATH%
```

Place or symlink Vaccine Ontology release **2026-06-12** where `vo_architecture.py` expects, or set:

```bash
set VO_OWL=path\to\vo.owl
```

Official release: https://purl.obolibrary.org/obo/vo/releases/2026-06-12/vo.owl

## Main entrypoints

```bash
set PYTHONPATH=%CD%\osean_kg;%PYTHONPATH%
python -u -m ctr.eval_table1_lock
python -u -m ctr.eval_setting_b
python -u -m ctr.eval_after_vo_fix
python -u -m ctr.ontology_gnn --lomo
```

Frozen paper numbers live under `osean_kg/ckpts/sea_ml_v14/ctr/optimize/`.

## Zenodo supporting data

See `zenodo/README_DATA.md` and `zenodo/UPLOAD.md`.  
Build / refresh the deposit zip, then upload manually on Zenodo (or use `upload_draft.py` with `ZENODO_TOKEN` to save a draft).

## Citation

Manuscript draft: keep the working draft in your paper workspace; this repo is the code + artifact companion. After Zenodo upload, replace the manuscript data-availability DOI placeholder with the assigned DOI.
