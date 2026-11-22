# muril_lora (LoRA) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `google/muril-base-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/238,150,662

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8239 | 0.8239 | 0.8237 | 0.7354 | 0.0589 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8163    0.8196    0.8179      1258
     Neutral     0.8212    0.7927    0.8067      1095
    Positive     0.8349    0.8599    0.8472      1106

    accuracy                         0.8239      3459
   macro avg     0.8241    0.8240    0.8239      3459
weighted avg     0.8238    0.8239    0.8237      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 8 | 0.4307 | 0.8171 | True |
| 9 | 0.4065 | 0.8163 | False |
| 10 | 0.3868 | 0.8153 | False |
