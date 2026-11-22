# indicbert_lora (LoRA) — final

seed: 44 | dataset: CMTE-NITANP | split: B | hf: `ai4bharat/IndicBERTv2-MLM-only`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/278,635,782

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8109 | 0.8104 | 0.8104 | 0.7164 | 0.0382 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7853    0.8315    0.8077      1258
     Neutral     0.8374    0.7434    0.7876      1095
    Positive     0.8182    0.8544    0.8359      1106

    accuracy                         0.8109      3459
   macro avg     0.8136    0.8098    0.8104      3459
weighted avg     0.8123    0.8109    0.8104      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7994 | 0.7815 | True |
| 2 | 0.5665 | 0.7864 | True |
| 3 | 0.5107 | 0.8035 | True |
| 4 | 0.4764 | 0.8032 | False |
| 5 | 0.4469 | 0.8034 | False |
