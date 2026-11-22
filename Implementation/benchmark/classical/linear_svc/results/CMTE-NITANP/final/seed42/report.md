# linear_svc (word11) — final

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'C': 0.1}` | frozen hash: `0cdd8027b0564b9b0f79bbc44f369242f2f4a3570d6664492d3dc6974a827410` | val macro-F1: 0.8096

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8103 | 0.8096 | 0.8100 | 0.7151 | 0.0328 |
| test | 0.8231 | 0.8231 | 0.8230 | 0.7340 | 0.0431 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7986    0.8386    0.8181      1258
     Neutral     0.8240    0.7826    0.8028      1095
    Positive     0.8515    0.8454    0.8485      1106

    accuracy                         0.8231      3459
   macro avg     0.8247    0.8222    0.8231      3459
weighted avg     0.8236    0.8231    0.8230      3459
```
