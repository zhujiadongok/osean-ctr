"""Create or update a Zenodo deposition draft (does NOT publish).

Requires env ZENODO_TOKEN with deposit:write scope.

Usage:
  # first time: creates a new draft
  python upload_draft.py

  # later: set DEPOSITION_ID to the draft id returned previously
  python upload_draft.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

# Set after first create, or leave 0 to create a new draft.
DEPOSITION_ID = 0
ZIP = Path(__file__).resolve().parent / "OSEAN-CTR_supporting_data.zip"
BASE = "https://zenodo.org/api"


def metadata() -> dict:
    return {
        "title": (
            "OSEAN-CTR supporting data: Vaccine Ontology, CTR feature "
            "sidecars, and frozen evaluation artifacts"
        ),
        "upload_type": "dataset",
        "description": (
            "<p>Supporting data for OSEAN-CTR (SEA-CDM–based vaccine trial "
            "evidence integration and ontology-guided design-field prediction). "
            "Includes Vaccine Ontology (release 2026-06-12) when packaged, CTR "
            "feature sidecars, frozen result JSON artifacts, mapping audit "
            "overlays, and class-label helpers. Companion code: "
            '<a href="https://github.com/zhujiadongok/osean-ctr">'
            "https://github.com/zhujiadongok/osean-ctr</a>. Full AACT dump is "
            "not redistributed; snapshot identifiers are in the manuscript "
            "Table S8.</p>"
        ),
        "creators": [
            {"name": "Zhu, Jiadong", "affiliation": "University of Michigan"},
            {"name": "Yeh, Feng-Yu", "affiliation": "University of Michigan"},
            {"name": "Zheng, Jie", "affiliation": "University of Michigan"},
            {"name": "He, Yongqun", "affiliation": "University of Michigan"},
        ],
        "keywords": [
            "clinical trials",
            "SEA-CDM",
            "Vaccine Ontology",
            "ClinicalTrials.gov",
            "knowledge graph",
            "OSEAN-CTR",
        ],
        "access_right": "open",
        "license": "cc-by-4.0",
        "related_identifiers": [
            {
                "identifier": "https://github.com/zhujiadongok/osean-ctr",
                "relation": "isSupplementTo",
                "resource_type": "software",
            },
            {
                "identifier": "http://purl.obolibrary.org/obo/vo/releases/2026-06-12/vo.owl",
                "relation": "isDerivedFrom",
                "resource_type": "dataset",
            },
        ],
    }


def main() -> None:
    token = os.environ.get("ZENODO_TOKEN") or os.environ.get("ZENODO_ACCESS_TOKEN")
    if not token:
        print("ERROR: set ZENODO_TOKEN (deposit:write). Not publishing.", file=sys.stderr)
        sys.exit(2)
    if not ZIP.is_file():
        print(f"ERROR: missing {ZIP}", file=sys.stderr)
        sys.exit(2)

    headers = {"Authorization": f"Bearer {token}"}
    dep_id = DEPOSITION_ID
    if not dep_id:
        r = requests.post(f"{BASE}/deposit/depositions", json={}, headers=headers, timeout=60)
        r.raise_for_status()
        dep = r.json()
        dep_id = dep["id"]
        print("created draft id:", dep_id)
    else:
        r = requests.get(f"{BASE}/deposit/depositions/{dep_id}", headers=headers, timeout=60)
        r.raise_for_status()
        dep = r.json()

    if dep.get("submitted"):
        print("ERROR: deposition already published; create a new version on Zenodo UI.", file=sys.stderr)
        sys.exit(3)

    for f in dep.get("files") or []:
        fid = f.get("id")
        if not fid:
            continue
        dr = requests.delete(
            f"{BASE}/deposit/depositions/{dep_id}/files/{fid}",
            headers=headers,
            timeout=60,
        )
        print("delete", f.get("filename"), dr.status_code)

    # refresh links
    r = requests.get(f"{BASE}/deposit/depositions/{dep_id}", headers=headers, timeout=60)
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]
    with ZIP.open("rb") as fh:
        put = requests.put(
            f"{bucket}/{ZIP.name}",
            data=fh,
            headers={**headers, "Content-Type": "application/octet-stream"},
            timeout=600,
        )
    print("upload", put.status_code)
    put.raise_for_status()

    pr = requests.put(
        f"{BASE}/deposit/depositions/{dep_id}",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps({"metadata": metadata()}),
        timeout=60,
    )
    print("metadata", pr.status_code)
    if pr.status_code >= 400:
        print(pr.text[:1000], file=sys.stderr)
        pr.raise_for_status()

    out = pr.json()
    print(
        json.dumps(
            {
                "id": out.get("id"),
                "state": out.get("state"),
                "submitted": out.get("submitted"),
                "doi": out.get("doi") or out.get("metadata", {}).get("prereserve_doi"),
                "html": out.get("links", {}).get("html"),
                "note": "Saved as draft only; publish on Zenodo UI when ready.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
