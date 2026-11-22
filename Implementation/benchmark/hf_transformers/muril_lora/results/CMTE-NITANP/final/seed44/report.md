# muril_lora (LoRA) — final

seed: 44 | dataset: CMTE-NITANP | split: B | hf: `google/muril-base-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/238,150,662

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8089 | 0.8081 | 0.8082 | 0.7134 | 0.0476 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7833    0.8362    0.8089      1258
     Neutral     0.8342    0.7352    0.7816      1095
    Positive     0.8175    0.8508    0.8339      1106

    accuracy                         0.8089      3459
   macro avg     0.8117    0.8074    0.8081      3459
weighted avg     0.8104    0.8089    0.8082      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 1.0120 | 0.4300 | True |
| 2 | 0.8804 | 0.7311 | True |
| 3 | 0.7037 | 0.7745 | True |
| 4 | 0.5998 | 0.7894 | True |
| 5 | 0.5438 | 0.7998 | True |
| 6 | 0.5101 | 0.8048 | True |
| 7 | 0.4820 | 0.8043 | False |
| 8 | 0.4608 | 0.8041 | False |
