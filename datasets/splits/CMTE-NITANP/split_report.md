# De-duplication & split report

Corpus: **CMTE-NITANP**, 34,588 rows | shingle=word3 | threshold=0.8 | num_perm=128 | seed=42 | ratios=0.8,0.1,0.1

## Phase 1 - clusters

| metric | value |
|---|---:|
| distinct `text_clean` | 33,353 |
| exact-duplicate groups | 987 |
| exact-duplicate rows (beyond first) | 1,235 |
| near-dup clusters (group_id) | 33,313 |
| rows in a cluster of size >1 | 2,283 (6.60%) |
| multi-row clusters with conflicting labels | 115 (11.41% of multi-row) |
| rows inside conflicting-label clusters | 314 |
| mean label entropy (multi-row clusters) | 0.113 bits |

Cluster-size distribution (size: clusters): 1:32,305, 2:906, 3:45, 4:30, 5:8, 6:5, 7:2, 8:2, 9:2, 10:2, 11:3, 14:1, 15:1, 16:1

## Phases 2/3 - splits

| split | rows | Negative | Neutral | Positive | test rows w/ twin in train |
|---|---:|---:|---:|---:|---:|
| A (random) / train | 27,670 | 10,066 | 8,755 | 8,849 | - |
| A (random) / val | 3,459 | 1,259 | 1,094 | 1,106 | - |
| A (random) / test | 3,459 | 1,258 | 1,095 | 1,106 | 203 (5.87%) |
| B (grouped) / train | 27,670 | 10,066 | 8,755 | 8,849 | - |
| B (grouped) / val | 3,459 | 1,259 | 1,094 | 1,106 | - |
| B (grouped) / test | 3,459 | 1,258 | 1,095 | 1,106 | 0 (0.00%) |

## Is Split B a genuine regroup or just Split A with twins moved?

Symmetric difference of row membership between A and B. `|A ^ B|` counts rows that
leave *and* enter a split, so it is even and reaches at most 2x the split size; the
middle column is the number of rows whose membership actually changed.
A small train value (~hundreds) would mean B is just A with the duplicate rows
relocated; a large value means B is an independent regroup.

| split | \|A ^ B\| rows | rows changed (\|A^B\|/2) | share of split |
|---|---:|---:|---:|
| train | 11,064 | 5,532 | 19.99% |
| val | 6,248 | 3,124 | 90.32% |
| test | 6,202 | 3,101 | 89.65% |

## Threshold sensitivity (shingle=word3)

| threshold | clusters | rows in cluster>1 | A test leak | B test leak |
|---|---:|---:|---:|---:|
| 0.7 | 33,277 | 2,348 (6.79%) | 6.10% | 0.00% |
| 0.8 | 33,313 | 2,283 (6.60%) | 5.87% | 0.00% |
| 0.9 | 33,349 | 2,227 (6.44%) | 5.81% | 0.00% |

Model evaluation (TF-IDF+SVM, CNN+BiLSTM) lives in `Implementation/baseline/`.
