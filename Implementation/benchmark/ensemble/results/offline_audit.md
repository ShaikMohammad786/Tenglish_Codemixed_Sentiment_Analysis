# Offline ensemble audit (validation-selected)

| ensemble | k | test acc | test macro-F1 |
|---|---:|---:|---:|
| val_top25 | 25 | 0.8500 | 0.8502 |
| val_top15 | 15 | 0.8500 | 0.8501 |
| fullft+linearsvc_union+logreg_union+muril_lora+soft_voting+linear_svc | 6 | 0.8500 | 0.8501 |
| val_top13 | 13 | 0.8491 | 0.8491 |
| val_top17 | 17 | 0.8485 | 0.8486 |
| val_top21 | 21 | 0.8479 | 0.8481 |
| val_top5 | 5 | 0.8479 | 0.8480 |
| val_top9 | 9 | 0.8479 | 0.8480 |
| val_top7 | 7 | 0.8474 | 0.8474 |
| fullft+linearsvc_union+logreg_union+muril_lora+softvote_union | 5 | 0.8468 | 0.8469 |
| val_top11 | 11 | 0.8468 | 0.8468 |
| val_top3 | 3 | 0.8465 | 0.8467 |
| all_saved | 28 | 0.8465 | 0.8466 |

## Individual models (validation rank)

| model | val F1 | test acc | test F1 |
|---|---:|---:|---:|
| logreg_union | 0.8278 | 0.8323 | 0.8324 |
| muril_fullft | 0.8255 | 0.8338 | 0.8336 |
| linearsvc_union | 0.8232 | 0.8364 | 0.8363 |
| logreg_union_norm | 0.8226 | 0.8309 | 0.8309 |
| logreg_char25 | 0.8221 | 0.8274 | 0.8276 |
| linearsvc_union_norm | 0.8191 | 0.8320 | 0.8320 |
| telugu_bert_lora | 0.8175 | 0.8147 | 0.8149 |
| muril_lora | 0.8171 | 0.8239 | 0.8239 |
| soft_voting | 0.8115 | 0.8225 | 0.8227 |
| distilmbert_lora | 0.8109 | 0.8077 | 0.8073 |
| softvote_union | 0.8100 | 0.8317 | 0.8318 |
| linear_svc | 0.8096 | 0.8231 | 0.8231 |
| logistic_regression | 0.8083 | 0.8291 | 0.8293 |
| mbert_lora | 0.8061 | 0.8063 | 0.8064 |
| indicbert_lora | 0.8060 | 0.8135 | 0.8134 |
| logreg_word11_norm | 0.8015 | 0.8187 | 0.8192 |
| extra_trees | 0.7999 | 0.8072 | 0.8073 |
| bilstm | 0.7996 | 0.8025 | 0.8026 |
| xlmr_lora | 0.7970 | 0.8025 | 0.8035 |
| lightgbm | 0.7952 | 0.8049 | 0.8048 |
| multinomial_nb | 0.7951 | 0.8130 | 0.8121 |
| hist_gradient_boosting | 0.7913 | 0.8020 | 0.8022 |
| textcnn | 0.7899 | 0.7892 | 0.7894 |
| random_forest | 0.7873 | 0.7956 | 0.7954 |
| cnn_bilstm | 0.7717 | 0.7826 | 0.7833 |
| xgboost | 0.7709 | 0.7774 | 0.7773 |
| knn | 0.7663 | 0.7759 | 0.7755 |
| dummy | 0.1779 | 0.3637 | 0.1778 |
