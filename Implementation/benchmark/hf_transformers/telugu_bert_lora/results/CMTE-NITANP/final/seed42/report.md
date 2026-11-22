# telugu_bert_lora (LoRA) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `l3cube-pune/telugu-bert`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/238,150,662

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8147 | 0.8149 | 0.8146 | 0.7215 | 0.0746 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8110    0.8052    0.8081      1258
     Neutral     0.8042    0.7991    0.8016      1095
    Positive     0.8289    0.8409    0.8348      1106

    accuracy                         0.8147      3459
   macro avg     0.8147    0.8151    0.8149      3459
weighted avg     0.8146    0.8147    0.8146      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.9728 | 0.6856 | True |
| 2 | 0.6915 | 0.7730 | True |
| 3 | 0.5730 | 0.7980 | True |
| 4 | 0.5227 | 0.8049 | True |
| 5 | 0.4895 | 0.8092 | True |
| 6 | 0.4615 | 0.8125 | True |
| 7 | 0.4416 | 0.8162 | True |
| 8 | 0.4214 | 0.8164 | True |
| 9 | 0.4073 | 0.8174 | True |
| 10 | 0.3953 | 0.8175 | True |
