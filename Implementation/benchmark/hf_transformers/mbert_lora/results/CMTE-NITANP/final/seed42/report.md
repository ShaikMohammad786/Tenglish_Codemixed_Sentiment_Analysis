# mbert_lora (LoRA) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `bert-base-multilingual-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/178,447,878

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8063 | 0.8064 | 0.8062 | 0.7088 | 0.0424 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7936    0.8100    0.8017      1258
     Neutral     0.7974    0.7763    0.7867      1095
    Positive     0.8296    0.8318    0.8307      1106

    accuracy                         0.8063      3459
   macro avg     0.8069    0.8060    0.8064      3459
weighted avg     0.8063    0.8063    0.8062      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7665 | 0.7670 | True |
| 2 | 0.5775 | 0.7906 | True |
| 3 | 0.5228 | 0.8061 | True |
| 4 | 0.4826 | 0.7941 | False |
| 5 | 0.4518 | 0.8041 | False |
