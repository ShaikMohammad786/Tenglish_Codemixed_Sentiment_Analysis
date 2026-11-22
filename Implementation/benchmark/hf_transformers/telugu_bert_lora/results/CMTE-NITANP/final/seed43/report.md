# telugu_bert_lora (LoRA) — final

seed: 43 | dataset: CMTE-NITANP | split: B | hf: `l3cube-pune/telugu-bert`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/238,150,662

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8205 | 0.8207 | 0.8205 | 0.7304 | 0.0616 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8242    0.8052    0.8146      1258
     Neutral     0.7942    0.8073    0.8007      1095
    Positive     0.8424    0.8508    0.8466      1106

    accuracy                         0.8205      3459
   macro avg     0.8203    0.8211    0.8207      3459
weighted avg     0.8206    0.8205    0.8205      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.9706 | 0.7107 | True |
| 2 | 0.6538 | 0.7925 | True |
| 3 | 0.5539 | 0.7993 | True |
| 4 | 0.5104 | 0.7991 | False |
| 5 | 0.4733 | 0.8079 | True |
| 6 | 0.4510 | 0.8092 | True |
| 7 | 0.4284 | 0.8042 | False |
| 8 | 0.4121 | 0.8132 | True |
| 9 | 0.3956 | 0.8119 | False |
| 10 | 0.3849 | 0.8102 | False |
