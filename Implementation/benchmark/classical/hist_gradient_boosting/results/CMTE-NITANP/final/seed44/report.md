# hist_gradient_boosting (word11) — final

seed: 44 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'learning_rate': 0.1, 'max_iter': 200}` | frozen hash: `c01d49bac162c1501077e68730a8be0a476f5937ff5872c465ee37a0596956fe` | val macro-F1: 0.7928

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7927 | 0.7928 | 0.7928 | 0.6883 | 0.0253 |
| test | 0.8023 | 0.8024 | 0.8022 | 0.7026 | 0.0219 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7884    0.8084    0.7983      1258
     Neutral     0.7927    0.7753    0.7839      1095
    Positive     0.8279    0.8219    0.8249      1106

    accuracy                         0.8023      3459
   macro avg     0.8030    0.8019    0.8024      3459
weighted avg     0.8024    0.8023    0.8022      3459
```
