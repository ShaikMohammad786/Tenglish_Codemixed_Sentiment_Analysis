# muril_fullft (full FT) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `google/muril-base-cased`

hyper: `{'lr': 2e-05, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 8, 'grad_accum': 4}` | frozen hash: `7a7a41d68eee1650ec376112a0e097374e3ad4350ce324e9c112ba06549a1223` | trainable: 237,558,531/237,558,531

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8338 | 0.8336 | 0.8336 | 0.7501 | 0.0702 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8273    0.8378    0.8325      1258
     Neutral     0.8250    0.8009    0.8128      1095
    Positive     0.8494    0.8617    0.8555      1106

    accuracy                         0.8338      3459
   macro avg     0.8339    0.8335    0.8336      3459
weighted avg     0.8336    0.8338    0.8336      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.8860 | 0.7847 | True |
| 2 | 0.5187 | 0.8108 | True |
| 3 | 0.4132 | 0.8202 | True |
| 4 | 0.3427 | 0.8255 | True |
| 5 | 0.2897 | 0.8227 | False |
| 6 | 0.2416 | 0.8241 | False |
