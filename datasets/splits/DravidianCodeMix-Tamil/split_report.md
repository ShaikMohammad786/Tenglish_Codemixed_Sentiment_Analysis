# De-duplication & split report

Corpus: **DravidianCodeMix-Tamil**, 44,020 rows | shingle=word3 | threshold=0.8 | num_perm=128 | seed=42 | ratios=0.8,0.1,0.1

## Phase 1 - clusters

| metric | value |
|---|---:|
| distinct `text_clean` | 43,560 |
| exact-duplicate groups | 342 |
| exact-duplicate rows (beyond first) | 460 |
| near-dup clusters (group_id) | 43,464 |
| rows in a cluster of size >1 | 932 (2.12%) |
| multi-row clusters with conflicting labels | 73 (19.41% of multi-row) |
| rows inside conflicting-label clusters | 225 |
| mean label entropy (multi-row clusters) | 0.182 bits |

Cluster-size distribution (size: clusters): 1:43,088, 2:294, 3:43, 4:20, 5:6, 6:3, 7:5, 8:2, 9:1, 12:1, 15:1

## Phases 2/3 - splits

| split | rows | Negative | Neutral | Positive | test rows w/ twin in train |
|---|---:|---:|---:|---:|---:|
| A (random) / train | 35,216 | 4,182 | 0 | 19,899 | - |
| A (random) / val | 4,402 | 523 | 0 | 2,487 | - |
| A (random) / test | 4,402 | 523 | 0 | 2,487 | 78 (1.77%) |
| B (grouped) / train | 35,216 | 4,182 | 0 | 19,898 | - |
| B (grouped) / val | 4,402 | 523 | 0 | 2,487 | - |
| B (grouped) / test | 4,402 | 523 | 0 | 2,488 | 0 (0.00%) |

## Is Split B a genuine regroup or just Split A with twins moved?

Symmetric difference of row membership between A and B. `|A ^ B|` counts rows that
leave *and* enter a split, so it is even and reaches at most 2x the split size; the
middle column is the number of rows whose membership actually changed.
A small train value (~hundreds) would mean B is just A with the duplicate rows
relocated; a large value means B is an independent regroup.

| split | \|A ^ B\| rows | rows changed (\|A^B\|/2) | share of split |
|---|---:|---:|---:|
| train | 14,104 | 7,052 | 20.02% |
| val | 7,952 | 3,976 | 90.32% |
| test | 7,906 | 3,953 | 89.80% |

## Threshold sensitivity (shingle=word3)

| threshold | clusters | rows in cluster>1 | A test leak | B test leak |
|---|---:|---:|---:|---:|
| 0.7 | 43,315 | 1,155 (2.62%) | 2.25% | 0.00% |
| 0.8 | 43,464 | 932 (2.12%) | 1.77% | 0.00% |
| 0.9 | 43,521 | 843 (1.92%) | 1.52% | 0.00% |

Model evaluation (TF-IDF+SVM, CNN+BiLSTM) lives in `Implementation/baseline/`.
