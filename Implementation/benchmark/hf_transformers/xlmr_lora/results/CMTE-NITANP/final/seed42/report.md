# xlmr_lora (LoRA) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `xlm-roberta-base`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 1,182,723/279,228,678

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8025 | 0.8035 | 0.8032 | 0.7037 | 0.0234 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7938    0.7957    0.7948      1258
     Neutral     0.7545    0.8027    0.7779      1095
    Positive     0.8674    0.8101    0.8378      1106

    accuracy                         0.8025      3459
   macro avg     0.8052    0.8029    0.8035      3459
weighted avg     0.8049    0.8025    0.8032      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7999 | 0.7680 | True |
| 2 | 0.6030 | 0.7861 | True |
| 3 | 0.5422 | 0.7970 | True |
| 4 | 0.5080 | 0.7873 | False |
| 5 | 0.4811 | 0.7968 | False |
