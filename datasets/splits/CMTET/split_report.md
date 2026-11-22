# De-duplication & split report

Corpus: **CMTET**, 19,868 rows | shingle=word3 | threshold=0.8 | num_perm=128 | seed=42 | ratios=0.8,0.1,0.1

## Phase 1 - clusters

| metric | value |
|---|---:|
| distinct `text_clean` | 19,719 |
| exact-duplicate groups | 78 |
| exact-duplicate rows (beyond first) | 149 |
| near-dup clusters (group_id) | 19,578 |
| rows in a cluster of size >1 | 402 (2.02%) |
| multi-row clusters with conflicting labels | 62 (55.36% of multi-row) |
| rows inside conflicting-label clusters | 199 |
| mean label entropy (multi-row clusters) | 0.634 bits |

Cluster-size distribution (size: clusters): 1:19,466, 2:60, 3:18, 4:19, 5:10, 7:1, 9:1, 11:1, 14:1, 61:1

## Phases 2/3 - splits

| split | rows | Negative | Neutral | Positive | test rows w/ twin in train |
|---|---:|---:|---:|---:|---:|
| A (random) / train | 15,894 | 6,173 | 3,378 | 6,343 | - |
| A (random) / val | 1,987 | 772 | 422 | 793 | - |
| A (random) / test | 1,987 | 772 | 422 | 793 | 44 (2.21%) |
| B (grouped) / train | 15,894 | 6,174 | 3,377 | 6,343 | - |
| B (grouped) / val | 1,987 | 772 | 422 | 793 | - |
| B (grouped) / test | 1,987 | 771 | 423 | 793 | 0 (0.00%) |

## Is Split B a genuine regroup or just Split A with twins moved?

Symmetric difference of row membership between A and B. `|A ^ B|` counts rows that
leave *and* enter a split, so it is even and reaches at most 2x the split size; the
middle column is the number of rows whose membership actually changed.
A small train value (~hundreds) would mean B is just A with the duplicate rows
relocated; a large value means B is an independent regroup.

| split | \|A ^ B\| rows | rows changed (\|A^B\|/2) | share of split |
|---|---:|---:|---:|
| train | 6,366 | 3,183 | 20.03% |
| val | 3,538 | 1,769 | 89.03% |
| test | 3,594 | 1,797 | 90.44% |

## Threshold sensitivity (shingle=word3)

| threshold | clusters | rows in cluster>1 | A test leak | B test leak |
|---|---:|---:|---:|---:|
| 0.7 | 19,536 | 465 (2.34%) | 2.42% | 0.00% |
| 0.8 | 19,578 | 402 (2.02%) | 2.21% | 0.00% |
| 0.9 | 19,666 | 310 (1.56%) | 1.66% | 0.00% |

Model evaluation (TF-IDF+SVM, CNN+BiLSTM) lives in `Implementation/baseline/`.
