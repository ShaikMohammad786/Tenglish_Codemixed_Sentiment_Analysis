# indicbert_lora (LoRA) — final

seed: 42 | dataset: CMTE-NITANP | split: B | hf: `ai4bharat/IndicBERTv2-MLM-only`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/278,635,782

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8135 | 0.8134 | 0.8131 | 0.7202 | 0.0624 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7818    0.8315    0.8059      1258
     Neutral     0.8396    0.7507    0.7927      1095
    Positive     0.8284    0.8553    0.8416      1106

    accuracy                         0.8135      3459
   macro avg     0.8166    0.8125    0.8134      3459
weighted avg     0.8150    0.8135    0.8131      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7796 | 0.7723 | True |
| 2 | 0.5616 | 0.7833 | True |
| 3 | 0.5086 | 0.7986 | True |
| 4 | 0.4737 | 0.8060 | True |
| 5 | 0.4467 | 0.7977 | False |
| 6 | 0.4195 | 0.8007 | False |
