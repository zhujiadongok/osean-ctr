"""Print current LOMO lines for S4 from after_vo_fix_eval.json."""
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "ckpts/sea_ml_v14/ctr/optimize/mapping_audit/after_vo_fix_eval.json"
d = json.loads(p.read_text(encoding="utf-8"))
rows = d["lomo_per_class"]
parts = []
for r in rows:
    delta = r["gated"] - r["exact"]
    parts.append(
        f"{r['vo'][-4:]} n={r['n']} exact {r['exact']:.3f} gated {r['gated']:.3f} (Δ{delta:+.3f})"
    )
print("; ".join(parts))
print("SUMMARY", d["lomo_summary"])
