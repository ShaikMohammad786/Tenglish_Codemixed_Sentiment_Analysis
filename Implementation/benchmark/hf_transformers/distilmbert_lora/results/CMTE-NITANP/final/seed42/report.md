# distilmbert_lora (LoRA) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `distilbert-base-multilingual-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 887,811/136,214,790

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8077 | 0.8073 | 0.8073 | 0.7121 | 0.0629 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8239    0.7886    0.8058      1258
     Neutral     0.8087    0.7680    0.7878      1095
    Positive     0.7909    0.8689    0.8281      1106

    accuracy                         0.8077      3459
   macro avg     0.8078    0.8085    0.8073      3459
weighted avg     0.8085    0.8077    0.8073      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7361 | 0.7758 | True |
| 2 | 0.5598 | 0.7884 | True |
| 3 | 0.5064 | 0.7959 | True |
| 4 | 0.4620 | 0.8050 | True |
| 5 | 0.4268 | 0.8099 | True |
| 6 | 0.3959 | 0.8109 | True |
| 7 | 0.3656 | 0.8075 | False |
| 8 | 0.3389 | 0.8058 | False |
