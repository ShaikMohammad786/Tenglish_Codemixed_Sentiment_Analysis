# distilmbert_lora (LoRA) — final

seed: 43 | dataset: CMTE-NITANP | split: B | hf: `distilbert-base-multilingual-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 887,811/136,214,790

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8086 | 0.8087 | 0.8083 | 0.7127 | 0.0497 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8078    0.7886    0.7981      1258
     Neutral     0.7991    0.7808    0.7898      1095
    Positive     0.8183    0.8590    0.8381      1106

    accuracy                         0.8086      3459
   macro avg     0.8084    0.8094    0.8087      3459
weighted avg     0.8084    0.8086    0.8083      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7330 | 0.7763 | True |
| 2 | 0.5605 | 0.7993 | True |
| 3 | 0.5034 | 0.7999 | True |
| 4 | 0.4607 | 0.8011 | True |
| 5 | 0.4250 | 0.8066 | True |
| 6 | 0.3937 | 0.8057 | False |
| 7 | 0.3667 | 0.8060 | False |
