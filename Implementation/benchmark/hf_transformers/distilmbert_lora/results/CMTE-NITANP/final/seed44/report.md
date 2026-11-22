# distilmbert_lora (LoRA) — final

seed: 44 | dataset: CMTE-NITANP | split: B | hf: `distilbert-base-multilingual-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 887,811/136,214,790

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8115 | 0.8115 | 0.8116 | 0.7166 | 0.0498 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8079    0.8124    0.8101      1258
     Neutral     0.7878    0.7900    0.7889      1095
    Positive     0.8394    0.8318    0.8356      1106

    accuracy                         0.8115      3459
   macro avg     0.8117    0.8114    0.8115      3459
weighted avg     0.8116    0.8115    0.8116      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7357 | 0.7600 | True |
| 2 | 0.5619 | 0.7896 | True |
| 3 | 0.5023 | 0.7969 | True |
| 4 | 0.4618 | 0.8010 | True |
| 5 | 0.4209 | 0.8039 | True |
| 6 | 0.3902 | 0.8071 | True |
| 7 | 0.3575 | 0.8053 | False |
| 8 | 0.3371 | 0.8052 | False |
