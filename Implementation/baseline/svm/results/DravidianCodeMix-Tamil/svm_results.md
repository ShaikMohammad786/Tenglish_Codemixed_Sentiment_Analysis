# Baseline: TF-IDF + LinearSVC

text field: `text_clean` | splits: `datasets/splits/DravidianCodeMix-Tamil` | seed: 42

| split | rows (train/val/test) | vocab | C | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| A | 35,216 / 4,402 / 4,402 | 14,598 | 1.0 | 0.4480 | 0.6170 | 0.4598 |
| B | 35,216 / 4,402 / 4,402 | 14,701 | 1.0 | 0.4420 | 0.6136 | 0.4515 |

**Leakage gap (A - B): macro-F1 +0.0083 | accuracy +0.0034**

C selected on val only; test never used for tuning.

## Per-class report

### Split A

```
                precision    recall  f1-score   support

Mixed_feelings     0.2659    0.1440    0.1868       493
      Negative     0.4317    0.3384    0.3794       523
      Positive     0.7030    0.8480    0.7687      2487
     not-Tamil     0.6644    0.4737    0.5531       209
 unknown_state     0.4514    0.3768    0.4107       690

      accuracy                         0.6170      4402
     macro avg     0.5033    0.4362    0.4598      4402
  weighted avg     0.5805    0.6170    0.5910      4402
```

### Split B

```
                precision    recall  f1-score   support

Mixed_feelings     0.2744    0.1481    0.1924       493
      Negative     0.4251    0.3365    0.3757       523
      Positive     0.7029    0.8509    0.7698      2488
     not-Tamil     0.6691    0.4471    0.5360       208
 unknown_state     0.4238    0.3507    0.3838       690

      accuracy                         0.6136      4402
     macro avg     0.4991    0.4267    0.4515      4402
  weighted avg     0.5765    0.6136    0.5868      4402
```
