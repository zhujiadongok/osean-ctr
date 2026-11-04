"""NVKG-style text recommendation scored like Devarakonda et al. JBI 2024 Table 6.

Task (aligned metrics, local vaccine CTR cohort — not their NCT IDs):
  Hold out test trials; recommend primary-endpoint titles / inclusion-criterion
  lines from neighbors; score max cosine similarity vs gold texts + rank/MRR.

Neighbor arms:
  - exact VO kNN (OSEAN matched weights)
  - OntSim+role kNN
  - title TF-IDF kNN (NVKG-style inductive simplification; no HRGAT/TrialTrove)

Embedding for text similarity: sklearn TF-IDF (word+char). Published NVKG used
fastText — numbers are same *protocol*, not identical model.
"""

from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctr import config as cfg  # noqa: E402
from ctr.build_table import build_ctr_table  # noqa: E402
from ctr.recommend import ontology_similarity_matrix  # noqa: E402

AACT_VAX = cfg.ROOT / "AACT Database" / "vaccine_nct_subset"
AACT_FULL = cfg.ROOT / "AACT Database" / "6wvqgoof4dut7eaxn610uzfajvnt"
OUT_DIR = cfg.CTR_DIR / "optimize" / "nvkg_style_text_recommend"
OUT_JSON = OUT_DIR / "nvkg_style_text_recommend.json"
OUT_MD = OUT_DIR / "REPORT.md"

# Match NVKG Table 6 Top-N settings
PEP_NS = (3, 5, 10)
ICR_NS = (10, 30, 50)
NEIGHBOR_POOL = 80  # pull enough neighbors to fill Top-N unique texts


def _read_pipe(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="|",
        dtype=str,
        low_memory=False,
        usecols=usecols,
    ).fillna("")


def load_titles(ncts: set[str]) -> dict[str, str]:
    stud = _read_pipe(
        AACT_VAX / "studies.txt",
        usecols=["nct_id", "brief_title", "official_title"],
    )
    stud = stud[stud["nct_id"].isin(ncts)]
    out: dict[str, str] = {}
    for _, r in stud.iterrows():
        t = str(r["brief_title"]).strip() or str(r["official_title"]).strip()
        out[str(r["nct_id"])] = t
    return out


def _chunk_filter(path: Path, ncts: set[str], usecols: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, sep="|", dtype=str, low_memory=False, usecols=usecols, chunksize=250_000
    ):
        c = chunk[chunk["nct_id"].isin(ncts)]
        if len(c):
            parts.append(c)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True).fillna("")


def load_primary_endpoints(ncts: set[str]) -> dict[str, list[str]]:
    """Prefer design_outcomes.measure (protocol); fallback outcomes.title."""
    out: dict[str, list[str]] = {}
    path = AACT_FULL / "design_outcomes.txt"
    if path.exists():
        d = _chunk_filter(
            path, ncts, ["nct_id", "outcome_type", "measure", "description"]
        )
        d = d[d["outcome_type"].str.lower() == "primary"]
        for nct, g in d.groupby("nct_id"):
            texts: list[str] = []
            seen: set[str] = set()
            for _, r in g.iterrows():
                t = str(r["measure"]).strip() or str(r["description"]).strip()
                s = re.sub(r"\s+", " ", t).strip()
                if len(s) < 8:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                texts.append(s)
            if texts:
                out[str(nct)] = texts
    if len(out) >= len(ncts) * 0.5:
        return out
    # fallback / fill gaps from results-module outcomes
    oc = _chunk_filter(
        AACT_FULL / "outcomes.txt", ncts, ["nct_id", "outcome_type", "title"]
    )
    oc = oc[oc["outcome_type"].str.upper() == "PRIMARY"]
    for nct, g in oc.groupby("nct_id"):
        if str(nct) in out:
            continue
        texts = []
        seen: set[str] = set()
        for t in g["title"].astype(str):
            s = re.sub(r"\s+", " ", t).strip()
            if len(s) < 8:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(s)
        if texts:
            out[str(nct)] = texts
    return out


_INCL_SPLIT = re.compile(
    r"inclusion\s+criteria\s*:?\s*",
    re.I,
)
_EXCL_SPLIT = re.compile(
    r"exclusion\s+criteria\s*:?\s*",
    re.I,
)


def _parse_inclusion_lines(criteria: str) -> list[str]:
    raw = str(criteria or "").replace("\r\n", "\n").replace("~", "\n")
    if not raw.strip():
        return []
    m_in = _INCL_SPLIT.search(raw)
    if m_in:
        rest = raw[m_in.end() :]
        m_ex = _EXCL_SPLIT.search(rest)
        block = rest[: m_ex.start()] if m_ex else rest
    else:
        # no labeled section — treat whole block cautiously
        block = raw
        if _EXCL_SPLIT.search(block) and not m_in:
            return []
    lines: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\n;]+|\*\s+|-\s+", block):
        s = re.sub(r"\s+", " ", part).strip(" .;:-")
        if len(s) < 12:
            continue
        if s.lower().startswith("exclusion"):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(s)
    return lines


def load_inclusion_criteria(ncts: set[str]) -> dict[str, list[str]]:
    # Prefer full AACT dump for completeness; fall back to vaccine subset.
    path = AACT_FULL / "eligibilities.txt"
    if not path.exists():
        path = AACT_VAX / "eligibilities.txt"
        el = _read_pipe(path, usecols=["nct_id", "criteria"])
        el = el[el["nct_id"].isin(ncts)]
    else:
        el = _chunk_filter(path, ncts, ["nct_id", "criteria"])
    out: dict[str, list[str]] = {}
    for _, r in el.iterrows():
        lines = _parse_inclusion_lines(r["criteria"])
        if lines:
            out[str(r["nct_id"])] = lines
    return out


class TextEncoder:
    """Prefer sentence-transformers; fall back to TF-IDF cosine."""

    def __init__(self, corpus: list[str]):
        self.kind = "tfidf"
        self._st = None
        self._vec: TfidfVectorizer | None = None
        self._cache: dict[str, np.ndarray] = {}
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.kind = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Using {self.kind}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"sentence-transformers unavailable ({exc}); using TF-IDF", flush=True)
            self._vec = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
                max_features=80_000,
            )
            self._vec.fit(corpus)
            self.kind = "tfidf-char_wb-3-5"
        # Pre-encode unique corpus strings once (large speedup for scoring)
        uniq = list(dict.fromkeys(c for c in corpus if c.strip()))
        print(f"Pre-encoding {len(uniq)} unique strings…", flush=True)
        self.prefetch(uniq)

    def prefetch(self, texts: list[str]) -> None:
        missing = [t for t in texts if t not in self._cache]
        if not missing:
            return
        if self._st is not None:
            emb = np.asarray(
                self._st.encode(
                    missing,
                    batch_size=128,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            for t, e in zip(missing, emb):
                self._cache[t] = e
            return
        assert self._vec is not None
        mat = self._vec.transform(missing)
        dense = mat.toarray().astype(np.float32)
        # L2 normalize for cosine via dot
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
        dense = dense / norms
        for t, e in zip(missing, dense):
            self._cache[t] = e

    def encode(self, texts: list[str]) -> np.ndarray:
        self.prefetch(texts)
        return np.stack([self._cache[t] for t in texts], axis=0)


def _cos_matrix(a, b) -> np.ndarray:
    """Dense or sparse cosine; rows of a vs rows of b."""
    if hasattr(a, "toarray") or hasattr(b, "toarray"):
        return cosine_similarity(a, b)
    # already L2-normalized MiniLM → dot product
    return np.asarray(a @ b.T, dtype=np.float64)


def unique_texts_from_neighbors(
    ranked: list[tuple[str, float]],
    bank: dict[str, list[str]],
    *,
    top_n: int,
    pool: int = NEIGHBOR_POOL,
) -> list[str]:
    """Collect unique texts from neighbors in rank order until top_n."""
    out: list[str] = []
    seen: set[str] = set()
    for nct, _s in ranked[:pool]:
        for t in bank.get(nct, []):
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= top_n:
                return out
    return out


def score_recommendations(
    gold_lists: dict[str, list[str]],
    recs_by_nct: dict[str, list[str]],
    enc: TextEncoder,
) -> dict[str, float]:
    """NVKG Table 6 style: per gold item, max sim to recs + rank of that rec."""
    ranks: list[float] = []
    recip: list[float] = []
    best_sims: list[float] = []
    n_queries = 0
    for nct, golds in gold_lists.items():
        recs = recs_by_nct.get(nct) or []
        if not golds or not recs:
            continue
        gX = enc.encode(golds)
        rX = enc.encode(recs)
        sims = _cos_matrix(gX, rX)
        for i in range(sims.shape[0]):
            n_queries += 1
            row = np.asarray(sims[i]).reshape(-1)
            j = int(np.argmax(row))
            best = float(row[j])
            rank = j + 1  # 1-based
            best_sims.append(best)
            ranks.append(float(rank))
            recip.append(1.0 / rank)
    if not best_sims:
        return {
            "n_queries": 0,
            "mean_rank": float("nan"),
            "mrr": float("nan"),
            "average_similarity": float("nan"),
        }
    return {
        "n_queries": int(n_queries),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(recip)),
        "average_similarity": float(np.mean(best_sims)),
    }


def title_knn_ranks(
    titles: dict[str, str],
    train_ids: list[str],
    test_ids: list[str],
    enc: TextEncoder,
) -> dict[str, list[tuple[str, float]]]:
    tr = [n for n in train_ids if titles.get(n)]
    te = [n for n in test_ids if titles.get(n)]
    if not tr or not te:
        return {}
    Xtr = enc.encode([titles[n] for n in tr])
    Xte = enc.encode([titles[n] for n in te])
    S = _cos_matrix(Xte, Xtr)
    out: dict[str, list[tuple[str, float]]] = {}
    for i, q in enumerate(te):
        row = np.asarray(S[i]).reshape(-1)
        order = np.argsort(-row)
        out[q] = [(tr[j], float(row[j])) for j in order if tr[j] != q]
    return out


def build_recs(
    ranked: dict[str, list[tuple[str, float]]],
    bank: dict[str, list[str]],
    test_ids: list[str],
    top_n: int,
) -> dict[str, list[str]]:
    return {
        q: unique_texts_from_neighbors(ranked.get(q, []), bank, top_n=top_n)
        for q in test_ids
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading CTR table…", flush=True)
    df = build_ctr_table()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    both = pd.concat([train, test], ignore_index=True)
    tr_ids = list(train["nct_id"])
    te_ids = list(test["nct_id"])
    ncts = set(tr_ids) | set(te_ids)
    print(f"CTR train={len(tr_ids)} test={len(te_ids)}", flush=True)

    print("Loading AACT titles / endpoints / inclusions…", flush=True)
    titles = load_titles(ncts)
    pep = load_primary_endpoints(ncts)
    icr = load_inclusion_criteria(ncts)
    print(
        f"titles={len(titles)} pep_ncts={len(pep)} icr_ncts={len(icr)}",
        flush=True,
    )

    corpus: list[str] = []
    corpus.extend(titles.values())
    for d in (pep, icr):
        for lst in d.values():
            corpus.extend(lst)
    corpus = [c for c in corpus if c.strip()]
    print(f"Building text encoder on {len(corpus)} strings…", flush=True)
    enc = TextEncoder(corpus)

    gold_pep = {n: pep[n] for n in te_ids if n in pep}
    gold_icr = {n: icr[n] for n in te_ids if n in icr}
    print(
        f"test with pep={len(gold_pep)} icr={len(gold_icr)}",
        flush=True,
    )

    print("OSEAN exact / OntSim+role neighbor ranks…", flush=True)
    ranked_exact = ontology_similarity_matrix(
        both, tr_ids, te_ids, mode="exact"
    )
    ranked_role = ontology_similarity_matrix(
        both, tr_ids, te_ids, mode="graph"
    )
    print("Title semantic kNN ranks…", flush=True)
    ranked_title = title_knn_ranks(titles, tr_ids, te_ids, enc)

    arms = OrderedDict(
        [
            ("osean_exact_knn", ranked_exact),
            ("osean_ontsim_role_knn", ranked_role),
            ("title_semantic_knn", ranked_title),
        ]
    )

    # Published NVKG Table 6 (HRGAT) for side-by-side reporting
    nvkg_published = {
        "primary_endpoints": {
            "3": {"average_similarity": 0.721, "mean_rank": 1.73, "mrr": 0.697},
            "5": {"average_similarity": 0.751, "mean_rank": 2.62, "mrr": 0.513},
            "10": {"average_similarity": 0.779, "mean_rank": 4.78, "mrr": 0.344},
        },
        "inclusion_criteria": {
            "10": {"average_similarity": 0.762, "mean_rank": 3.42, "mrr": 0.441},
            "30": {"average_similarity": 0.812, "mean_rank": 11.9, "mrr": 0.174},
            "50": {"average_similarity": 0.829, "mean_rank": 21.4, "mrr": 0.120},
        },
        "note": "Devarakonda et al. JBI 2024 Table 6 HRGAT; different cohort/IDs/fastText",
    }

    results: dict = {
        "protocol": {
            "cohort": "SEA-CDM vaccine CTR temporal split",
            "split": f"train start_date < {cfg.SPLIT_DATE}; test >=",
            "n_train": len(tr_ids),
            "n_test": len(te_ids),
            "n_test_with_pep": len(gold_pep),
            "n_test_with_icr": len(gold_icr),
            "text_similarity": enc.kind,
            "nvkg_similarity": "fastText (published; not re-run here)",
            "primary_source": "AACT design_outcomes.measure (primary)",
            "caveat": (
                "Same scoring protocol as NVKG Table 6 (max sim + rank/MRR), "
                "but vaccine CTR cohort and embedding model ≠ their drug-disease "
                "subset + fastText; they also cluster near-duplicate texts into shared nodes."
            ),
        },
        "nvkg_published_hrgat": nvkg_published,
        "ours": {},
    }

    for arm_name, ranked in arms.items():
        print(f"Scoring arm={arm_name}…", flush=True)
        arm_out: dict = {"primary_endpoints": {}, "inclusion_criteria": {}}
        for n in PEP_NS:
            recs = build_recs(ranked, pep, list(gold_pep.keys()), n)
            arm_out["primary_endpoints"][str(n)] = score_recommendations(
                gold_pep, recs, enc
            )
        for n in ICR_NS:
            recs = build_recs(ranked, icr, list(gold_icr.keys()), n)
            arm_out["inclusion_criteria"][str(n)] = score_recommendations(
                gold_icr, recs, enc
            )
        results["ours"][arm_name] = arm_out
        # quick print
        pep10 = arm_out["primary_endpoints"].get("10", {})
        icr10 = arm_out["inclusion_criteria"].get("10", {})
        print(
            f"  pep@10 avg_sim={pep10.get('average_similarity', float('nan')):.3f} "
            f"mrr={pep10.get('mrr', float('nan')):.3f} | "
            f"icr@10 avg_sim={icr10.get('average_similarity', float('nan')):.3f} "
            f"mrr={icr10.get('mrr', float('nan')):.3f}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_md(results)
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"wrote {OUT_MD}", flush=True)


def _write_md(results: dict) -> None:
    lines = [
        "# NVKG-style text recommendation (local vaccine CTR)",
        "",
        results["protocol"]["caveat"],
        "",
        f"- train={results['protocol']['n_train']} test={results['protocol']['n_test']}",
        f"- test with primary endpoints={results['protocol']['n_test_with_pep']}",
        f"- test with inclusion criteria={results['protocol']['n_test_with_icr']}",
        f"- similarity: {results['protocol']['text_similarity']}",
        "",
        "## Published NVKG (HRGAT, Table 6) — reference only",
        "",
        "| Task | Top-N | avg sim | mean rank | MRR |",
        "|------|------:|--------:|----------:|----:|",
    ]
    pub = results["nvkg_published_hrgat"]
    for task, key in (
        ("primary_endpoints", "Primary EP"),
        ("inclusion_criteria", "Inclusion"),
    ):
        for n, row in pub[task].items():
            lines.append(
                f"| {key} | {n} | {row['average_similarity']:.3f} | "
                f"{row['mean_rank']:.2f} | {row['mrr']:.3f} |"
            )
    lines += ["", "## Our methods (same metric definitions)", ""]
    for arm, blob in results["ours"].items():
        lines += [f"### {arm}", "", "| Task | Top-N | avg sim | mean rank | MRR | n_queries |", "|------|------:|--------:|----------:|----:|---------:|"]
        for task, label in (
            ("primary_endpoints", "Primary EP"),
            ("inclusion_criteria", "Inclusion"),
        ):
            for n, row in blob[task].items():
                lines.append(
                    f"| {label} | {n} | {row['average_similarity']:.3f} | "
                    f"{row['mean_rank']:.2f} | {row['mrr']:.3f} | {row['n_queries']} |"
                )
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
