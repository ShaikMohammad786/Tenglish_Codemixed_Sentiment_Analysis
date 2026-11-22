# electra_base (full FT) — final

seed: 42 | dataset: CMTE-NITANP-paper2 | split: A | hf: `/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/models/electra-base-discriminator`

hyper: `{'lr': 3e-05, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 32, 'epochs': 5, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `693ee24441a42dfa0e868c48c58e5da05195750cf94a775623a13277d99eead8` | trainable: 109,484,547/109,484,547

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8233 | 0.8231 | 0.8229 | 0.7350 | 0.0773 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8323    0.8114    0.8217      2397
     Neutral     0.8155    0.7887    0.8019      2158
    Positive     0.8211    0.8719    0.8457      2116

    accuracy                         0.8233      6671
   macro avg     0.8230    0.8240    0.8231      6671
weighted avg     0.8233    0.8233    0.8229      6671
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7310 | 0.7850 | True |
| 2 | 0.5049 | 0.7938 | True |
| 3 | 0.4004 | 0.8108 | True |
| 4 | 0.3191 | 0.8040 | False |
| 5 | 0.2586 | 0.7985 | False |
