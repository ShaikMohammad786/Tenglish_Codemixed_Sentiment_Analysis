# muril_fullft (full FT) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `google/muril-base-cased`

hyper: `{'lr': 2e-05, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 8, 'grad_accum': 4}` | frozen hash: `7a7a41d68eee1650ec376112a0e097374e3ad4350ce324e9c112ba06549a1223` | trainable: 85,647,363/237,558,531

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8271 | 0.8275 | 0.8269 | 0.7412 | 0.0456 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8446    0.7822    0.8122      1258
     Neutral     0.7925    0.8338    0.8126      1095
    Positive     0.8441    0.8716    0.8577      1106

    accuracy                         0.8271      3459
   macro avg     0.8271    0.8292    0.8275      3459
weighted avg     0.8280    0.8271    0.8269      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.4509 | 0.8231 | True |
| 2 | 0.4232 | 0.8195 | False |
