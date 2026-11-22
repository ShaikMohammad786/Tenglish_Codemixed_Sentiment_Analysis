# electra_base (full FT) — final

seed: 42 | dataset: CMTE-NITANP-paper2 | split: B | hf: `/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/models/electra-base-discriminator`

hyper: `{'lr': 3e-05, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 32, 'epochs': 5, 'batch_size': 16, 'grad_accum': 2}` | frozen hash: `693ee24441a42dfa0e868c48c58e5da05195750cf94a775623a13277d99eead8` | trainable: 109,484,547/109,484,547

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8069 | 0.8069 | 0.8067 | 0.7103 | 0.1169 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8009    0.8124    0.8066      2367
     Neutral     0.8127    0.7725    0.7921      2185
    Positive     0.8080    0.8362    0.8219      2119

    accuracy                         0.8069      6671
   macro avg     0.8072    0.8071    0.8069      6671
weighted avg     0.8070    0.8069    0.8067      6671
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7286 | 0.7909 | True |
| 2 | 0.5051 | 0.8117 | True |
| 3 | 0.4031 | 0.8051 | False |
| 4 | 0.3201 | 0.8174 | True |
| 5 | 0.2599 | 0.8174 | True |
