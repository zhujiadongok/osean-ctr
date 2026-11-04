"""Ontology-aware supervised GNN for CTR design-field prediction.

Builds the Study–feature–VO input graph (targets never appear as edges), trains a
relational message-passing encoder with multi-head classifiers, and reports:

  • chronological Setting A (train <2022 / test ≥2022)
  • fair ablation: +VO edges (subClassOf / has_role) vs no-VO
  • baselines: Exact kNN, OntSim+role kNN, HistGBDT, frequency
  • optional LOMO over the same 23 ontology-connected Materials as the paper

Inductive test encoding: query Study nodes are *not* required in the training
graph; we build a vector from linked VO / box features after message passing
(same pattern as ``supervised_gnn``).

Usage
-----
  # full chrono + ablation + baselines (~CPU tens of minutes)
  python -u -m ctr.ontology_gnn

  # smoke test (few epochs, skip LOMO)
  python -u -m ctr.ontology_gnn --quick

  # also run LOMO (expensive: one train per Material × 2 ablations)
  python -u -m ctr.ontology_gnn --lomo

  # LOMO only on first N eligible classes
  python -u -m ctr.ontology_gnn --lomo --lomo-max 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctr import config as cfg  # noqa: E402
from ctr.build_table import build_ctr_table  # noqa: E402
from ctr.eval_graph_value import (  # noqa: E402
    _eligible_ontology_connected,
    _score_bundle,
    _trial_mean_accuracy,
)
from ctr.graph import build_input_graph  # noqa: E402
from ctr.metrics import score_predictions  # noqa: E402
from ctr.recommend import ontology_knn_recommend  # noqa: E402
from ctr.regress import frequency_predict, regress_predict  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
OUT = cfg.CTR_DIR / "optimize" / "ontology_gnn"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class OntologyRGCN(torch.nn.Module):
    """Relational message-passing encoder + multi-target study heads."""

    def __init__(
        self,
        n_nodes: int,
        n_rel: int,
        dim: int,
        n_classes: dict[str, int],
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.x = torch.nn.Parameter(torch.empty(n_nodes, dim))
        torch.nn.init.xavier_uniform_(self.x)
        self.rel_w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(n_rel + 1, dim)) for _ in range(n_layers)]
        )
        for w in self.rel_w:
            torch.nn.init.xavier_uniform_(w)
        self.heads = torch.nn.ModuleDict(
            {
                t: torch.nn.Sequential(
                    torch.nn.Linear(dim, dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(dim, c),
                )
                for t, c in n_classes.items()
            }
        )

    def encode(self, src, rel, dst, deg) -> torch.Tensor:
        h = self.x
        for i in range(self.n_layers):
            w = self.rel_w[i]
            nbr = h[dst] * w[rel]
            nbr = nbr / deg[src].unsqueeze(1) / deg[dst].unsqueeze(1)
            msg = torch.zeros_like(h)
            msg.index_add_(0, src, nbr)
            h = F.relu(h + msg)
            if self.training and self.dropout > 0:
                h = F.dropout(h, p=self.dropout)
        return F.normalize(h, p=2, dim=1)

    def forward_studies(self, h: torch.Tensor, study_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        z = h[study_idx]
        return {t: head(z) for t, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_maps(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps = {}
    for t in STRUCT:
        labs = sorted(
            {str(x) for x in train[t].astype(str) if str(x) not in {"unknown", "nan", ""}}
        )
        maps[t] = {lab: i for i, lab in enumerate(labs)}
    return maps


def _y_tensor(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> dict[str, torch.Tensor]:
    out = {}
    for t, m in maps.items():
        counts = Counter(str(x) for x in df[t] if str(x) in m)
        maj = counts.most_common(1)[0][0] if counts else next(iter(m))
        idx = [m.get(str(v), m[maj]) for v in df[t].astype(str)]
        out[t] = torch.tensor(idx, dtype=torch.long)
    return out


def _edge_tensors(graph: dict, n: int):
    edges = torch.tensor(graph["triples"], dtype=torch.long)
    if len(edges) == 0:
        # degenerate: self-loops only
        loops = torch.stack(
            [torch.arange(n), torch.full((n,), graph["n_rel"], dtype=torch.long), torch.arange(n)],
            dim=1,
        )
        src, rel, dst = loops[:, 0], loops[:, 1], loops[:, 2]
        deg = torch.ones(n)
        return loops, src, rel, dst, deg
    both = torch.cat([edges, edges[:, [2, 1, 0]]], dim=0)
    loops = torch.stack(
        [torch.arange(n), torch.full((n,), graph["n_rel"], dtype=torch.long), torch.arange(n)],
        dim=1,
    )
    all_e = torch.cat([both, loops], dim=0)
    src, rel, dst = all_e[:, 0], all_e[:, 1], all_e[:, 2]
    deg = torch.bincount(src, minlength=n).float().clamp(min=1.0).sqrt()
    return edges, src, rel, dst, deg


def _inductive_vectors(
    h: np.ndarray,
    graph: dict,
    studies: pd.DataFrame,
    query_ids: list[str],
) -> np.ndarray:
    """Compose a query Study vector from VO + box feature node embeddings."""
    index = graph["index"]
    meta = studies.set_index("nct_id")
    wmap = {
        "arch_vo": 3,
        "disease": 2,
        "vo_platform": 2,
        "phase": 2,
        "primary_purpose": 1,
        "route": 1,
        "dose": 1,
        "sex": 1,
        "older": 1,
        "child": 1,
        "sponsor_class": 1,
    }
    cols = [
        ("arch_vo", None),
        ("disease", "DISEASE"),
        ("phase", "PHASE"),
        ("primary_purpose", "PURPOSE"),
        ("vo_platform", "PLATFORM"),
        ("route", "ROUTE"),
        ("dose", "DOSE"),
        ("sex", "SEX"),
        ("older", "OLDER"),
        ("child", "CHILD"),
        ("sponsor_class", "SPONSOR"),
    ]
    out = []
    for q in query_ids:
        row = meta.loc[q]
        parts = []
        vo = str(row["arch_vo"])
        if vo in index:
            parts.extend([h[index[vo]]] * wmap["arch_vo"])
        for col, prefix in cols[1:]:
            key = f"{prefix}:{row[col]}"
            if key in index:
                parts.extend([h[index[key]]] * wmap.get(col, 1))
        if not parts:
            out.append(np.zeros(h.shape[1], dtype=np.float32))
        else:
            v = np.mean(np.stack(parts), axis=0)
            out.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(out).astype(np.float32)


def _field(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_target": {t: rep["targets"][t] for t in STRUCT},
    }


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------


def train_and_predict(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    both: pd.DataFrame,
    *,
    include_vo_relations: bool = True,
    include_same_context: bool = True,
    dim: int = 128,
    epochs: int = 300,
    lr: float = 1e-2,
    lambda_link: float = 0.15,
    n_layers: int = 3,
    seed: int = 7,
    device: str | None = None,
) -> tuple[dict[str, dict[str, str]], dict]:
    """Train on ``fit`` graph; predict ``query`` inductively. Returns preds + meta."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    g = build_input_graph(
        fit,
        include_vo_relations=include_vo_relations,
        include_same_context=include_same_context,
    )
    n = len(g["nodes"])
    maps = _label_maps(fit)
    n_classes = {t: len(m) for t, m in maps.items()}
    inv = {t: {i: lab for lab, i in m.items()} for t, m in maps.items()}

    model = OntologyRGCN(n, g["n_rel"], dim, n_classes, n_layers=n_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    edges, src, rel, dst, deg = _edge_tensors(g, n)
    src, rel, dst, deg = src.to(dev), rel.to(dev), dst.to(dev), deg.to(dev)
    edges = edges.to(dev)

    study_ids = list(fit["nct_id"])
    study_idx = torch.tensor(
        [g["index"][f"STUDY:{s}"] for s in study_ids], dtype=torch.long, device=dev
    )
    y = {t: ten.to(dev) for t, ten in _y_tensor(fit, maps).items()}

    model.train()
    last_loss = 0.0
    for _ep in range(epochs):
        h = model.encode(src, rel, dst, deg)
        logits = model.forward_studies(h, study_idx)
        loss_cls = sum(F.cross_entropy(logits[t], y[t]) for t in STRUCT) / len(STRUCT)
        if len(edges) > 0:
            perm = edges[torch.randperm(len(edges), device=dev)][: min(1024, len(edges))]
            hh, tt = perm[:, 0], perm[:, 2]
            neg = torch.randint(0, n, (len(perm),), device=dev)
            pos = (h[hh] - h[tt]).pow(2).sum(1)
            neg_s = (h[hh] - h[neg]).pow(2).sum(1)
            loss_link = F.relu(pos - neg_s + 1.0).mean()
        else:
            loss_link = torch.tensor(0.0, device=dev)
        loss = loss_cls + lambda_link * loss_link
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        h_np = model.encode(src, rel, dst, deg).cpu().numpy()
        qids = list(query["nct_id"])
        qv = _inductive_vectors(h_np, g, both, qids)
        qt = torch.tensor(qv, dtype=torch.float32, device=dev)
        preds: dict[str, dict[str, str]] = {q: {} for q in qids}
        for t, head in model.heads.items():
            pred_i = head(qt).argmax(dim=1).tolist()
            for q, i in zip(qids, pred_i):
                preds[q][t] = inv[t].get(int(i), "unknown")

    meta = {
        "include_vo_relations": include_vo_relations,
        "include_same_context": include_same_context,
        "n_nodes": n,
        "n_triples": len(g["triples"]),
        "n_vo_relation_edges": g.get("n_vo_relation_edges", 0),
        "n_same_context_edges": g.get("n_same_context_edges", 0),
        "dim": dim,
        "epochs": epochs,
        "lr": lr,
        "lambda_link": lambda_link,
        "n_layers": n_layers,
        "final_loss": last_loss,
        "device": device,
        "n_fit": len(fit),
        "n_query": len(query),
    }
    return preds, meta


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def run_chrono(df: pd.DataFrame, train_cfg: dict, *, skip_baselines: bool = False) -> dict:
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    both = pd.concat([train, test], ignore_index=True)
    print(f"[chrono] train={len(train)} test={len(test)}", flush=True)

    results: dict = {"n_train": len(train), "n_test": len(test), "methods": {}}
    preds_store: dict = {}

    # GNN ablations
    for name, vo_flag in (
        ("ontology_rgcn_+vo", True),
        ("ontology_rgcn_no_vo", False),
    ):
        t0 = time.time()
        print(f"  train {name}…", flush=True)
        pred, meta = train_and_predict(
            train,
            test,
            both,
            include_vo_relations=vo_flag,
            **train_cfg,
        )
        sc = _field(score_predictions(test, pred, STRUCT))
        sc["trial_mean_accuracy"] = _trial_mean_accuracy(test, pred)
        sc["train_meta"] = meta
        sc["seconds"] = round(time.time() - t0, 1)
        results["methods"][name] = sc
        preds_store[name] = pred
        print(
            f"    field={sc['mean_accuracy']:.4f} trial={sc['trial_mean_accuracy']:.4f} "
            f"vo_edges={meta['n_vo_relation_edges']} ({sc['seconds']}s)",
            flush=True,
        )

    vo = results["methods"]["ontology_rgcn_+vo"]["mean_accuracy"]
    novo = results["methods"]["ontology_rgcn_no_vo"]["mean_accuracy"]
    results["delta_vo_minus_novo_field"] = float(vo - novo)

    if not skip_baselines:
        baselines = {
            "exact_knn@20": lambda: ontology_knn_recommend(
                both, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20, mode="exact"
            ),
            "ontsim_role_knn@20": lambda: ontology_knn_recommend(
                both, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20, mode="graph"
            ),
            "hist_gbdt": lambda: regress_predict(train, test, "gbdt", STRUCT),
            "frequency_by_phase": lambda: frequency_predict(train, test, STRUCT),
        }
        for name, fn in baselines.items():
            print(f"  baseline {name}…", flush=True)
            pred = fn()
            sc = _field(score_predictions(test, pred, STRUCT))
            sc["trial_mean_accuracy"] = _trial_mean_accuracy(test, pred)
            results["methods"][name] = sc
            preds_store[name] = pred
            print(f"    field={sc['mean_accuracy']:.4f} trial={sc['trial_mean_accuracy']:.4f}", flush=True)

    return results, preds_store


def run_lomo(
    df: pd.DataFrame,
    train_cfg: dict,
    *,
    max_classes: int | None = None,
    skip_baselines: bool = False,
) -> dict:
    eligible = _eligible_ontology_connected(df)
    if max_classes is not None:
        eligible = eligible[:max_classes]
    print(f"[lomo] classes={len(eligible)}", flush=True)

    rows = []
    for i, mat in enumerate(eligible, 1):
        te = df[df["arch_vo"].astype(str) == mat].reset_index(drop=True)
        tr = df[df["arch_vo"].astype(str) != mat].reset_index(drop=True)
        both = pd.concat([tr, te], ignore_index=True)
        row: dict = {"material": mat, "n": len(te)}
        print(f"  [{i}/{len(eligible)}] {mat} n={len(te)}", flush=True)

        for name, vo_flag in (
            ("ontology_rgcn_+vo", True),
            ("ontology_rgcn_no_vo", False),
        ):
            pred, meta = train_and_predict(
                tr,
                te,
                both,
                include_vo_relations=vo_flag,
                **train_cfg,
            )
            sc = _score_bundle(te, pred)
            row[name] = sc["trial_mean_accuracy"]
            row[f"{name}_field"] = sc["field_mean_accuracy"]
            row[f"{name}_vo_edges"] = meta["n_vo_relation_edges"]

        if not skip_baselines:
            for bname, mode in (("exact_knn@20", "exact"), ("ontsim_role_knn@20", "graph")):
                pred = ontology_knn_recommend(
                    both, list(tr["nct_id"]), list(te["nct_id"]), STRUCT, k=20, mode=mode
                )
                row[bname] = _trial_mean_accuracy(te, pred)
            row["hist_gbdt"] = _trial_mean_accuracy(
                te, regress_predict(tr, te, "gbdt", STRUCT)
            )

        row["delta_vo"] = float(row["ontology_rgcn_+vo"] - row["ontology_rgcn_no_vo"])
        rows.append(row)
        print(
            f"    +vo={row['ontology_rgcn_+vo']:.4f} no_vo={row['ontology_rgcn_no_vo']:.4f} "
            f"Δ={row['delta_vo']:+.4f}",
            flush=True,
        )

    num_keys = [
        "ontology_rgcn_+vo",
        "ontology_rgcn_no_vo",
        "delta_vo",
    ]
    for extra in ("exact_knn@20", "ontsim_role_knn@20", "hist_gbdt"):
        if extra in rows[0]:
            num_keys.append(extra)

    n_tot = sum(r["n"] for r in rows)
    summary = {}
    for k in num_keys:
        vals = [r[k] for r in rows]
        summary[f"{k}_class"] = float(np.mean(vals))
        summary[f"{k}_trial"] = float(sum(r[k] * r["n"] for r in rows) / n_tot)

    return {"eligible": eligible, "per_class": rows, "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ontology-aware supervised GNN for CTR")
    ap.add_argument("--quick", action="store_true", help="Few epochs; skip LOMO; for smoke tests")
    ap.add_argument("--lomo", action="store_true", help="Also run leave-one-Material-out")
    ap.add_argument("--lomo-max", type=int, default=None, help="Cap LOMO classes (debug)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="Load ckpts/.../ctr/ctr_studies.csv instead of rebuilding from SEA-CDM (Great Lakes friendly)",
    )
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    if args.quick:
        train_cfg = dict(
            dim=args.dim or 64,
            epochs=args.epochs or 60,
            lr=args.lr,
            lambda_link=0.15,
            n_layers=args.layers,
            seed=args.seed,
            device=args.device,
        )
        do_lomo = False
    else:
        train_cfg = dict(
            dim=args.dim or 128,
            epochs=args.epochs or 300,
            lr=args.lr,
            lambda_link=0.15,
            n_layers=args.layers,
            seed=args.seed,
            device=args.device,
        )
        do_lomo = args.lomo

    print("=== Ontology-aware CTR GNN ===", flush=True)
    print(f"train_cfg={train_cfg}", flush=True)
    print(f"device={train_cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)

    cache_csv = cfg.CTR_DIR / "ctr_studies.csv"
    if args.from_cache or not cfg.PROFILE.exists():
        if not cache_csv.exists():
            raise SystemExit(f"missing CTR cache: {cache_csv}")
        print(f"loading cohort from cache {cache_csv}", flush=True)
        df = pd.read_csv(cache_csv, dtype=str).fillna("")
        if "split" not in df.columns and "start_date" in df.columns:
            df["start_date_parsed"] = pd.to_datetime(df["start_date"], errors="coerce")
            df["split"] = np.where(
                df["start_date_parsed"] >= pd.Timestamp(cfg.SPLIT_DATE), "test", "train"
            )
    else:
        df = build_ctr_table()
    print(f"cohort={len(df)}", flush=True)

    chrono, _preds = run_chrono(df, train_cfg, skip_baselines=args.no_baselines)

    payload: dict = {
        "description": (
            "Supervised relational GNN on Study–feature–VO input graph. "
            "Ablation toggles VO subClassOf/has_role edges only. "
            "Test studies encoded inductively from feature/VO embeddings."
        ),
        "train_cfg": train_cfg,
        "chrono": chrono,
        "quick": bool(args.quick),
    }

    if do_lomo:
        # LOMO uses fewer epochs by default unless user set --epochs
        lomo_cfg = dict(train_cfg)
        if args.epochs is None and not args.quick:
            lomo_cfg["epochs"] = min(train_cfg["epochs"], 150)
        payload["lomo"] = run_lomo(
            df,
            lomo_cfg,
            max_classes=args.lomo_max,
            skip_baselines=args.no_baselines,
        )

    payload["seconds_total"] = round(time.time() - t_all, 1)
    out_json = OUT / "ontology_gnn_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable report
    lines = [
        "# Ontology-aware CTR GNN",
        "",
        f"train_cfg: {json.dumps(train_cfg)}",
        f"total_seconds: {payload['seconds_total']}",
        "",
        "## Chronological Setting A (field-mean accuracy)",
    ]
    for name, sc in sorted(
        chrono["methods"].items(), key=lambda kv: -kv[1]["mean_accuracy"]
    ):
        lines.append(
            f"  {sc['mean_accuracy']:.4f}  trial={sc.get('trial_mean_accuracy', float('nan')):.4f}  {name}"
        )
    lines.append(f"  Δ(+vo − no_vo) field = {chrono.get('delta_vo_minus_novo_field', 0):+.4f}")
    if "lomo" in payload:
        lines.append("")
        lines.append("## LOMO summary (trial-mean / class-mean)")
        for k, v in sorted(payload["lomo"]["summary"].items()):
            lines.append(f"  {k}: {v:.4f}")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
