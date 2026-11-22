# telugu_bert_lora (LoRA) — final

seed: 44 | dataset: CMTE-NITANP | split: B | hf: `l3cube-pune/telugu-bert`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/238,150,662

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8228 | 0.8233 | 0.8230 | 0.7336 | 0.0539 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8102    0.8211    0.8156      1258
     Neutral     0.7955    0.8100    0.8027      1095
    Positive     0.8662    0.8373    0.8515      1106

    accuracy                         0.8228      3459
   macro avg     0.8240    0.8228    0.8233      3459
weighted avg     0.8235    0.8228    0.8230      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.9755 | 0.7049 | True |
| 2 | 0.6690 | 0.7896 | True |
| 3 | 0.5629 | 0.8004 | True |
| 4 | 0.5147 | 0.8022 | True |
| 5 | 0.4791 | 0.8135 | True |
| 6 | 0.4547 | 0.8135 | True |
| 7 | 0.4352 | 0.8201 | True |
| 8 | 0.4167 | 0.8162 | False |
| 9 | 0.4010 | 0.8168 | False |
