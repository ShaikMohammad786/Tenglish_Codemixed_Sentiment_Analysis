# muril_lora (LoRA) — final

seed: 43 | dataset: CMTE-NITANP | split: B | hf: `google/muril-base-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/238,150,662

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8173 | 0.8174 | 0.8174 | 0.7253 | 0.0556 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7977    0.8339    0.8154      1258
     Neutral     0.8043    0.7918    0.7980      1095
    Positive     0.8546    0.8237    0.8389      1106

    accuracy                         0.8173      3459
   macro avg     0.8189    0.8164    0.8174      3459
weighted avg     0.8180    0.8173    0.8174      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 1.0076 | 0.4977 | True |
| 2 | 0.7975 | 0.7627 | True |
| 3 | 0.6157 | 0.7915 | True |
| 4 | 0.5509 | 0.8009 | True |
| 5 | 0.5099 | 0.8035 | True |
| 6 | 0.4833 | 0.8074 | True |
| 7 | 0.4594 | 0.8074 | False |
| 8 | 0.4396 | 0.8122 | True |
| 9 | 0.4228 | 0.8105 | False |
| 10 | 0.4136 | 0.8093 | False |
