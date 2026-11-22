# hist_gradient_boosting (word11) — final

seed: 46 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'learning_rate': 0.1, 'max_iter': 200}` | frozen hash: `c01d49bac162c1501077e68730a8be0a476f5937ff5872c465ee37a0596956fe` | val macro-F1: 0.7895

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7892 | 0.7895 | 0.7893 | 0.6831 | 0.0223 |
| test | 0.7994 | 0.7995 | 0.7993 | 0.6983 | 0.0224 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7809    0.8076    0.7941      1258
     Neutral     0.7919    0.7644    0.7779      1095
    Positive     0.8283    0.8246    0.8265      1106

    accuracy                         0.7994      3459
   macro avg     0.8004    0.7989    0.7995      3459
weighted avg     0.7996    0.7994    0.7993      3459
```
