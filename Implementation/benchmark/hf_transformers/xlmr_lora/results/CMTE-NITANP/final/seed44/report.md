# xlmr_lora (LoRA) — final

seed: 44 | dataset: CMTE-NITANP | split: B | hf: `xlm-roberta-base`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 1,182,723/279,228,678

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8170 | 0.8167 | 0.8166 | 0.7254 | 0.0784 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8072    0.8156    0.8114      1258
     Neutral     0.8352    0.7680    0.8002      1095
    Positive     0.8120    0.8671    0.8387      1106

    accuracy                         0.8170      3459
   macro avg     0.8181    0.8169    0.8167      3459
weighted avg     0.8176    0.8170    0.8166      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.8035 | 0.7505 | True |
| 2 | 0.6005 | 0.7856 | True |
| 3 | 0.5443 | 0.7850 | False |
| 4 | 0.5032 | 0.7953 | True |
| 5 | 0.4733 | 0.8023 | True |
| 6 | 0.4555 | 0.7993 | False |
| 7 | 0.4320 | 0.8030 | True |
| 8 | 0.4119 | 0.8071 | True |
| 9 | 0.3968 | 0.8078 | True |
| 10 | 0.3785 | 0.8063 | False |
