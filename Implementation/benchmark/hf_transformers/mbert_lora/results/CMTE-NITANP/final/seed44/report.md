# mbert_lora (LoRA) — final

seed: 44 | dataset: CMTE-NITANP | split: B | hf: `bert-base-multilingual-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/178,447,878

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8144 | 0.8147 | 0.8145 | 0.7209 | 0.0879 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8003    0.8156    0.8079      1258
     Neutral     0.7930    0.7909    0.7920      1095
    Positive     0.8525    0.8363    0.8444      1106

    accuracy                         0.8144      3459
   macro avg     0.8153    0.8143    0.8147      3459
weighted avg     0.8147    0.8144    0.8145      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7906 | 0.7627 | True |
| 2 | 0.5836 | 0.7814 | True |
| 3 | 0.5269 | 0.7933 | True |
| 4 | 0.4855 | 0.7953 | True |
| 5 | 0.4544 | 0.8047 | True |
| 6 | 0.4262 | 0.8078 | True |
| 7 | 0.4002 | 0.8107 | True |
| 8 | 0.3838 | 0.8066 | False |
| 9 | 0.3615 | 0.8108 | True |
| 10 | 0.3506 | 0.8081 | False |
