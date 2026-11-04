import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctr.build_table import build_ctr_table
from ctr.eval_beat_nvkg_angles import apply_cluster_bank, cluster_texts
from ctr.eval_nvkg_style_text_recommend import (
    TextEncoder,
    build_recs,
    load_inclusion_criteria,
    load_primary_endpoints,
    load_titles,
    score_recommendations,
    title_knn_ranks,
)
from ctr.recommend import ontology_similarity_matrix

df = build_ctr_table()
train = df[df.split == "train"]
test = df[df.split == "test"]
both = pd.concat([train, test], ignore_index=True)
tr, te = list(train.nct_id), list(test.nct_id)
ncts = set(tr) | set(te)
titles = load_titles(ncts)
pep = load_primary_endpoints(ncts)
icr = load_inclusion_criteria(ncts)
corpus = list(titles.values())
for d in (pep, icr):
    for lst in d.values():
        corpus.extend(lst)
enc = TextEncoder([c for c in corpus if c.strip()])
ranked_title = title_knn_ranks(titles, tr, te, enc)
ranked_role = ontology_similarity_matrix(both, tr, te, mode="graph")
all_pep = [t for lst in pep.values() for t in lst]
all_icr = [t for lst in icr.values() for t in lst]
for thr in (0.68, 0.65, 0.62, 0.60):
    mp = cluster_texts(all_pep, enc, thr)
    mi = cluster_texts(all_icr, enc, thr)
    pep_c = apply_cluster_bank(pep, mp)
    icr_c = apply_cluster_bank(icr, mi)
    gold_pep = {n: pep_c[n] for n in te if n in pep_c}
    gold_icr = {n: icr_c[n] for n in te if n in icr_c}
    for name, ranked in (("title", ranked_title), ("ontsim", ranked_role)):
        sp = score_recommendations(
            gold_pep, build_recs(ranked, pep_c, list(gold_pep), 10), enc
        )
        si10 = score_recommendations(
            gold_icr, build_recs(ranked, icr_c, list(gold_icr), 10), enc
        )
        si50 = score_recommendations(
            gold_icr, build_recs(ranked, icr_c, list(gold_icr), 50), enc
        )
        print(
            f"thr={thr} {name}: pep10={sp['average_similarity']:.3f} "
            f"icr10={si10['average_similarity']:.3f} "
            f"icr50={si50['average_similarity']:.3f} "
            f"pep_c={len(set(mp.values()))} icr_c={len(set(mi.values()))}",
            flush=True,
        )
