import json
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "ckpts" / "sea_ml_v14" / "ctr" / "optimize"
t = json.loads((base / "table1_lock.json").read_text(encoding="utf-8"))
b = json.loads((base / "setting_b" / "setting_b.json").read_text(encoding="utf-8"))
for label, payload, key in (
    ("TABLE1", t, "test_scores"),
    ("SETTINGB", b, "test_scores"),
):
    print(label)
    for name, s in sorted(payload[key].items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  {s['mean_accuracy']:.4f}  F1 {s['mean_macro_f1']:.4f}  {name}")
print("boot_b", b["bootstrap_ontsim_vs_exact"]["ontsim_role_minus_exact"])
