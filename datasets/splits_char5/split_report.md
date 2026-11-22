# De-duplication & split report

Corpus: **CMTE-NITANP**, 34,588 rows | shingle=char5 | threshold=0.8 | num_perm=128 | seed=42 | ratios=0.8,0.1,0.1

Held out entirely (not split, not seen in train/val/test): **CMTET** (19,868 rows)

## Phase 1 - clusters

| metric | value |
|---|---:|
| distinct `text_clean` | 33,353 |
| exact-duplicate groups | 987 |
| exact-duplicate rows (beyond first) | 1,235 |
| near-dup clusters (group_id) | 33,216 |
| rows in a cluster of size >1 | 2,434 (7.04%) |
| multi-row clusters with conflicting labels | 123 (11.58% of multi-row) |
| rows inside conflicting-label clusters | 350 |
| mean label entropy (multi-row clusters) | 0.114 bits |

Cluster-size distribution (size: clusters): 1:32,154, 2:938, 3:62, 4:33, 5:10, 6:3, 7:2, 8:3, 9:2, 10:2, 11:2, 12:1, 14:1, 15:1, 16:1, 17:1

## Phases 2/3 - splits

| split | rows | Negative | Neutral | Positive | test rows w/ twin in train |
|---|---:|---:|---:|---:|---:|
| A (random) / train | 27,670 | 10,066 | 8,755 | 8,849 | - |
| A (random) / val | 3,459 | 1,259 | 1,094 | 1,106 | - |
| A (random) / test | 3,459 | 1,258 | 1,095 | 1,106 | 217 (6.27%) |
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
| train | 11,010 | 5,505 | 19.90% |
| val | 6,182 | 3,091 | 89.36% |
| test | 6,272 | 3,136 | 90.66% |

## Threshold sensitivity (shingle=char5)

| threshold | clusters | rows in cluster>1 | A test leak | B test leak |
|---|---:|---:|---:|---:|
| 0.7 | 32,985 | 2,776 (8.03%) | 7.08% | 0.00% |
| 0.8 | 33,216 | 2,434 (7.04%) | 6.27% | 0.00% |
| 0.9 | 33,313 | 2,287 (6.61%) | 5.93% | 0.00% |

Model evaluation (TF-IDF+SVM, CNN+BiLSTM) lives in `Implementation/baseline/`.
