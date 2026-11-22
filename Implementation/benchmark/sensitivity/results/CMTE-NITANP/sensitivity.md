# Transformer hyperparameter sensitivity (val only, fixed epoch budget)

Seed 42 | epochs 5 | dataset CMTE-NITANP

| model | config | lr | max_len | batch | eff. batch | best epoch | val macro-F1 | val acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| muril_lora | baseline_main | 0.0004 | 64 | 16 | 32 | 8 | 0.8171 | -- |
| muril_lora | lr2e4_len64_bs16 | 0.0002 | 64 | 16 | 32 | 5 | 0.7693 | 0.7713 |
| muril_lora | lr4e4_len96_bs16 | 0.0004 | 96 | 16 | 32 | 5 | 0.7805 | 0.7812 |
| muril_lora | lr4e4_len64_bs8 | 0.0004 | 64 | 8 | 32 | 5 | 0.7986 | 0.7994 |
| mbert_lora | baseline_main | 0.0004 | 64 | 16 | 32 | 3 | 0.8061 | -- |
| mbert_lora | lr2e4_len64_bs16 | 0.0002 | 64 | 16 | 32 | 5 | 0.7976 | 0.7979 |
| mbert_lora | lr4e4_len96_bs16 | 0.0004 | 96 | 16 | 32 | 5 | 0.8022 | 0.8028 |
| mbert_lora | lr4e4_len64_bs8 | 0.0004 | 64 | 8 | 32 | 5 | 0.7971 | 0.7979 |
| distilmbert_lora | baseline_main | 0.0004 | 64 | 16 | 32 | 6 | 0.8109 | -- |
| distilmbert_lora | lr2e4_len64_bs16 | 0.0002 | 64 | 16 | 32 | 5 | 0.8008 | 0.8011 |
| distilmbert_lora | lr4e4_len96_bs16 | 0.0004 | 96 | 16 | 32 | 5 | 0.8092 | 0.8098 |
| distilmbert_lora | lr4e4_len64_bs8 | 0.0004 | 64 | 8 | 32 | 5 | 0.8097 | 0.8103 |
