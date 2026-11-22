# indicbert_lora (LoRA) — final

seed: 43 | dataset: CMTE-NITANP | split: B | hf: `ai4bharat/IndicBERTv2-MLM-only`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/278,635,782

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8150 | 0.8154 | 0.8151 | 0.7219 | 0.0761 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8073    0.8092    0.8083      1258
     Neutral     0.7905    0.8027    0.7966      1095
    Positive     0.8490    0.8336    0.8412      1106

    accuracy                         0.8150      3459
   macro avg     0.8156    0.8152    0.8154      3459
weighted avg     0.8153    0.8150    0.8151      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7809 | 0.7810 | True |
| 2 | 0.5615 | 0.7884 | True |
| 3 | 0.5093 | 0.8043 | True |
| 4 | 0.4762 | 0.8096 | True |
| 5 | 0.4453 | 0.8046 | False |
| 6 | 0.4219 | 0.8105 | True |
| 7 | 0.4036 | 0.8066 | False |
| 8 | 0.3842 | 0.8131 | True |
| 9 | 0.3670 | 0.8081 | False |
| 10 | 0.3523 | 0.8101 | False |
