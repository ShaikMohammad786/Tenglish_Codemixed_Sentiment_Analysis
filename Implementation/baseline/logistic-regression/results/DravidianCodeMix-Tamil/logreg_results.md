# Baseline: TF-IDF + Logistic Regression

text field: `text_clean` | splits: `datasets/splits/DravidianCodeMix-Tamil` | seed: 42

| split | rows (train/val/test) | vocab | C | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| A | 35,216 / 4,402 / 4,402 | 14,598 | 10.0 | 0.4477 | 0.6068 | 0.4627 |
| B | 35,216 / 4,402 / 4,402 | 14,701 | 10.0 | 0.4500 | 0.6020 | 0.4433 |

**Leakage gap (A - B): macro-F1 +0.0194 | accuracy +0.0048**

C selected on val only; test never used for tuning.

## Per-class report

### Split A

```
                precision    recall  f1-score   support

Mixed_feelings     0.2708    0.1846    0.2195       493
      Negative     0.4306    0.3442    0.3826       523
      Positive     0.7052    0.8187    0.7577      2487
     not-Tamil     0.6273    0.4833    0.5459       209
 unknown_state     0.4383    0.3812    0.4078       690

      accuracy                         0.6068      4402
     macro avg     0.4945    0.4424    0.4627      4402
  weighted avg     0.5784    0.6068    0.5880      4402
```

### Split B

```
                precision    recall  f1-score   support

Mixed_feelings     0.2508    0.1663    0.2000       493
      Negative     0.4033    0.3270    0.3611       523
      Positive     0.7097    0.8292    0.7648      2488
     not-Tamil     0.6294    0.4327    0.5128       208
 unknown_state     0.4060    0.3536    0.3780       690

      accuracy                         0.6020      4402
     macro avg     0.4798    0.4218    0.4433      4402
  weighted avg     0.5705    0.6020    0.5810      4402
```
