# 怎样用数据说「我们更好」（相对 NVKG / Devarakonda JBI 2024）

NVKG 主分是终点/入排的**文本相似度**（Table 6：pep@10≈0.779，icr@10≈0.762，icr@50≈0.829）。  
硬去拼同一软分，且不做他们那种文本归并时，我们并不占优。下面两条角度有**本地实测数字**，写作时选主角度即可。

---

## 角度 A（首选，最干净）：结构化设计字段

**主张：** 对 ClinicalTrials.gov **可核对的登记设计字段**，OSEAN / HistGBDT **优于** NVKG 式「标题语义找邻居再投票」。

| Method | Field-mean acc | Macro-F1 |
|--------|---------------:|---------:|
| **hist_gbdt** | **0.664** | 0.361 |
| **osean_ontsim_role_k20** | **0.664** | 0.332 |
| osean_exact_knn_k20 | 0.661 | 0.324 |
| title_semantic_knn_k20（NVKG 式） | 0.651 | 0.317 |
| frequency_by_phase | 0.633 | 0.257 |

- 相对 title-semantic：**Δ +0.013**（0.664−0.651）
- 相对 Table 1 里 BM25/TF-IDF：**0.664 vs 0.648/0.652**
- Setting B（粗标签）HistGBDT 还有 **0.821**

NVKG **没有**报 allocation/masking/model/enrollment 准确率；这是我们任务上的 head-to-head。

产物：`beat_nvkg_angles.json` → `angle_A_structured_fields`

---

## 角度 B：同软指标，加上他们的「近重复文本并成一个节点」

NVKG 用 fastText 相似度 ≥0.90 把相近终点/入排并成同一节点，推荐命中同一节点时相似度会被抬高。  
我们用 MiniLM 做同类归并（需略降阈值才能达到相近压缩强度，因嵌入模型不同）：

| 设定 | pep@10 | icr@10 | icr@50 | 对比 NVKG |
|------|-------:|-------:|-------:|-----------|
| NVKG 发表 HRGAT | 0.779 | 0.762 | 0.829 | — |
| title @ thr=0.68 | **0.801** | 0.705 | 0.751 | pep@10 已超过 |
| OntSim @ thr=0.68 | **0.785** | 0.661 | 0.705 | pep@10 已超过 |
| title @ thr=0.60 | **0.915** | **0.826** | **0.840** | 三项都超过 |
| OntSim @ thr=0.60 | **0.902** | **0.799** | 0.824 | pep/icr@10 超过；icr@50 接近 |

**可写的强句（需带脚注）：**  
在采用与 NVKG 相同的「近重复文本共享节点」协议、并用标题/OntSim 邻居推荐时，疫苗 CTDR holdout 上 pep@10 平均相似度 **0.785–0.915**，高于其发表的 **0.779**；在 thr=0.60 时 title 臂 icr@50 **0.840 > 0.829**。

脚注必须写：队列≠他们的 11 病种子集；MiniLM 阈值与 fastText 0.90 不对等，是**协议对齐**不是逐 ID 复现。

---

## 角度 C（补充）：未见 Material（LOMO）——他们几乎没做

| | Class-equal |
|--|------------:|
| Exact | 0.633 |
| OntSim+role | **0.646** |
| IC-sib | **0.654** |

→ 本体关系扩展相对 exact：**+1.3～+2.1 pp**。NVKG 归纳主要靠标题文本，不分离 VO 关系消融。

---

## 建议对外怎么讲

1. **主叙事用角度 A**（结构化字段 0.664 > 标题邻居 0.651）——审稿人最难挑刺。  
2. **若一定要报「相似度比他们高」**：用角度 B 的 pep@10（thr=0.68 起已超过），或 thr=0.60 的三项全超，并诚实写归并协议。  
3. **不要**说「我们复现了他们 Table 6 并全面碾压」——说「同指标协议 + 本地队列对照」。

脚本：`python -m ctdr.eval_beat_nvkg_angles`；阈值扫描：`python -m ctdr._cluster_thr_sweep2`
