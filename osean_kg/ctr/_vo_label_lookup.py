"""Print VO ids matching label needles."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vo_architecture import load_classes

needles = [n.lower() for n in sys.argv[1:]] or [
    "inactivated polio",
    "hepatitis a",
    "varicella",
    "zoster",
    "dendritic",
    "mrna",
    "tumor cell",
    "measles",
    "rotavirus",
    "japanese encephalitis",
]
classes = load_classes()
for needle in needles:
    hits = []
    for vid, meta in classes.items():
        if meta.get("deprecated"):
            continue
        lab = (meta.get("label") or "").lower()
        if needle in lab:
            hits.append((vid, meta.get("label")))
    print(f"== {needle} ({len(hits)})")
    for vid, lab in sorted(hits)[:20]:
        print(f"  {vid}\t{lab}")
