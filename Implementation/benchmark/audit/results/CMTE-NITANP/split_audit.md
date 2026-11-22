# Split audit — exact Jaccard verification

Dataset: **CMTE-NITANP** | threshold=0.8 | num_perm=128 | 2026-10-19T12:11:14+00:00

> `exact_pairs_ge_threshold` is the number of (test, train) pairs with exact word-3gram
> Jaccard >= threshold. Non-zero means the MinHash-LSH grouping missed real leakage.

| split | n_train | n_test | LSH candidates | exact >= θ | borderline 0.7–θ | max exact Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| A | 27,670 | 3,459 | 347 | 345 | 2 | 1.0 |
| B | 27,670 | 3,459 | 0 | 0 | 0 | 0.0 |

**Interpretation.** Split B is considered leakage-audited only if `exact_pairs_ge_threshold == 0`. Otherwise rebuild the split with an exact-Jaccard pass.
