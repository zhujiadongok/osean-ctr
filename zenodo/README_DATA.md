# OSEAN-CTR supporting data (Zenodo package)

Companion to code repository: https://github.com/zhujiadongok/osean-ctr

## Upload (manual)

1. Open https://zenodo.org/deposit/new (sign in).
2. Upload `OSEAN-CTR_supporting_data.zip` from this folder.
3. Paste title / description / creators / keywords from `package/ZENODO_METADATA.txt` (or the form filled by `upload_draft.py`).
4. Save as draft, review, then **Publish** when ready.
5. Copy the DOI into the manuscript data-availability statement.

Optional API draft update (does **not** publish): set `ZENODO_TOKEN` and run `python upload_draft.py`.

## Contents

| Path | Description |
|------|-------------|
| `vo/vo.owl` | Vaccine Ontology release **2026-06-12** (local copy used in analyses), if present in package |
| `sidecars/disease_vo_map.csv` | NCT → Material VO mapping used by CTR |
| `sidecars/study_context_profile.csv` | Design / phase / enrollment fields |
| `sidecars/study_intervention_profile.csv` | Route / dose |
| `sidecars/subject_demographics_summary.csv` | Eligibility sex / age flags |
| `artifacts/` | Frozen paper result JSONs (Table 1, Setting B, LOMO, mapping audit, …) |
| `labels/` | Class-label helpers when included |

## Not included

- Full AACT pipe dump (large). Local snapshot identifiers and checksums: manuscript Table S8.
- Full SEA-CDM official core table dump (only CTR sidecars above).

## Related

- Code: https://github.com/zhujiadongok/osean-ctr  
- VO official release IRI: http://purl.obolibrary.org/obo/vo/releases/2026-06-12/vo.owl  

## License

Code follows the GitHub repository. Ontology follows VO / OBO terms. Registry-derived tables are redistributed for research reproducibility; redistributors should respect ClinicalTrials.gov / AACT terms.
