# How to upload this package to Zenodo

You do **not** need the reserved DOI `10.5281/zenodo.23773934` (it 404s). Create a **new** deposit and use the DOI Zenodo assigns.

## Manual (recommended)

1. Go to https://zenodo.org/deposit/new and sign in.
2. Drag/drop `OSEAN-CTR_supporting_data.zip`.
3. Fill metadata from `package/ZENODO_METADATA.txt`.
4. Save draft → review files → **Publish**.
5. Copy `https://doi.org/10.5281/zenodo.XXXXXXXX` into the paper (title page + Data availability).

## Optional API draft

```bash
set ZENODO_TOKEN=...   # deposit:write scope
python upload_draft.py
```

`upload_draft.py` creates/updates a **draft** only (never publishes). After first successful draft, set `DEPOSITION_ID` in that script to the returned id.
