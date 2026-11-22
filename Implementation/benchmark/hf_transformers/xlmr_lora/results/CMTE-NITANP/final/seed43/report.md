# xlmr_lora (LoRA) — final

seed: 43 | dataset: CMTE-NITANP | split: B | hf: `xlm-roberta-base`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 1,182,723/279,228,678

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8109 | 0.8106 | 0.8104 | 0.7164 | 0.0625 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7975    0.8140    0.8057      1258
     Neutral     0.8330    0.7562    0.7927      1095
    Positive     0.8069    0.8617    0.8334      1106

    accuracy                         0.8109      3459
   macro avg     0.8125    0.8106    0.8106      3459
weighted avg     0.8118    0.8109    0.8104      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.8039 | 0.7728 | True |
| 2 | 0.5947 | 0.7835 | True |
| 3 | 0.5393 | 0.7883 | True |
| 4 | 0.5042 | 0.7875 | False |
| 5 | 0.4747 | 0.8006 | True |
| 6 | 0.4538 | 0.8098 | True |
| 7 | 0.4303 | 0.7993 | False |
| 8 | 0.4081 | 0.8108 | True |
| 9 | 0.3966 | 0.8101 | False |
| 10 | 0.3807 | 0.8083 | False |
