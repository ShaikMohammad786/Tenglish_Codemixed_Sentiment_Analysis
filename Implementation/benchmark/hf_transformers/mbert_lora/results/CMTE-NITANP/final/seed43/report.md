# mbert_lora (LoRA) — final

seed: 43 | dataset: CMTE-NITANP | split: B | hf: `bert-base-multilingual-cased`

hyper: `{'lr': 0.0004, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 64, 'epochs': 10, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `950ea13b4c2977714077f01143b823e287a2d0335f89d265a76943c7850eab24` | trainable: 592,131/178,447,878

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8089 | 0.8084 | 0.8084 | 0.7132 | 0.0668 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7844    0.8331    0.8080      1258
     Neutral     0.8301    0.7452    0.7854      1095
    Positive     0.8193    0.8445    0.8317      1106

    accuracy                         0.8089      3459
   macro avg     0.8113    0.8076    0.8084      3459
weighted avg     0.8100    0.8089    0.8084      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7826 | 0.7709 | True |
| 2 | 0.5854 | 0.7897 | True |
| 3 | 0.5238 | 0.7919 | True |
| 4 | 0.4825 | 0.7975 | True |
| 5 | 0.4541 | 0.7948 | False |
| 6 | 0.4245 | 0.8080 | True |
| 7 | 0.4062 | 0.8111 | True |
| 8 | 0.3829 | 0.8059 | False |
| 9 | 0.3640 | 0.8043 | False |
