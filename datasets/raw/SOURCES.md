# Raw dataset sources

Downloaded 2026-10-10 into `datasets/raw/`. All files are the upstream releases, fetched
unmodified via `raw.githubusercontent.com`. Cite the original authors in any publication.

## CMTE-NITANP — Telugu-English code-mixed sentiment (34k)

- **Origin:** Rayala, U. R. & Seshadri, K. *Sentiment analysis of code-mixed Telugu–English text
  using transformers.* Journal of Computational Social Science 9, 2 (2026).
  https://doi.org/10.1007/s42001-025-00434-3
- **Source repo:** https://github.com/rayalaupendar/Code-Mixed-Tel-Eng-34k

| File | Rows | Notes |
|---|---|---|
| `CMTE-NITANP/34k_Code_Mixed_Tel-Eng.csv` | 34,588 (+header) | columns `comment,label,` (trailing empty column); CRLF line endings |
| `CMTE-NITANP/dict_1pld.csv` | 9,388 | variant→canonical normalization pairs (no header); no trailing column |

Label scheme: `-1` = negative (12,583), `0` = neutral (10,944), `1` = positive (11,061).

## CMTET — Code-Mixed Telugu-English Text (sentiment + word-level LID)

- **Origin:** Kusampudi, S. S. V., Sathineni, P. & Mamidi, R. *Sentiment Analysis in Code-Mixed
  Telugu-English Text with Unsupervised Data Normalization.* RANLP 2021, pp. 753–760.
  https://aclanthology.org/2021.ranlp-1.86/
- **Source repos:** https://github.com/ksubbu199/cmtet-sentiment ·
  https://github.com/ksubbu199/cmtet-lid

| File | Content |
|---|---|
| `CMTET/codemix_sentiment_data.txt` | 19,868 annotated sentences (POS 7,929 / NEG 7,717 / NTL 4,222) with per-word LID tags; paper reports 19,857 |
| `CMTET/lid/twitter_{train,test}_data.txt` | Twitter LID split (70/30) |
| `CMTET/lid/blog_{train,test}_data.txt` | Blog LID split (70/30) |

### `codemix_sentiment_data.txt` format
Each record is two lines followed by a blank-line pair (`\n\n`), i.e. four lines per record:
1. `POS:`/`NEG:`/`NTL:` followed by the code-mixed sentence.
2. Space-separated per-word tags drawn from `en`, `te`, `ne`, `univ`.

### `CMTET/lid/*.txt` format
One `word tag` per line (single space, last field is the tag); a blank line marks the end of a sentence.

## DravidianCodeMix — Tamil-English code-mixed sentiment (+ offensive language)

- **Origin:** Chakravarthi, B. R., Priyadharshini, R., Muralidaran, V., Jose, N., Suryawanshi, S.,
  Sherly, E. & McCrae, J. P. *DravidianCodeMix: sentiment analysis and offensive language
  identification dataset for Dravidian languages in code-mixed text.* Language Resources and
  Evaluation (2022). https://doi.org/10.1007/s10579-022-09583-7
- **Source repo:** https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset (also Zenodo 4750858)
- **Bundle:** `DravidianCodeMix-Tamil/DravidianCodeMix-2020.zip`, extracted to
  `DravidianCodeMix-Tamil/DravidianCodeMix/` (bundle also contains Malayalam + Kannada files).

Tamil-English sentiment (5-class):

| File | Rows | Separator | Column order |
|---|---:|---|---|
| `DravidianCodeMix/tamil_sentiment_full_train.csv` | 35,220 | TAB | `text<TAB>label` |
| `DravidianCodeMix/tamil_sentiment_full_dev.csv` | 4,398 | `;` | `text;label;` |
| `DravidianCodeMix/tamil_sentiment_full_test.csv` | 4,402 | `;` | `text;label;` |
| `DravidianCodeMix/tamil_sentiment_full.csv` | 44,160 | TAB | **`label<TAB>text`** (reversed order) |

Also present: `tamil_offensive_full{,_train,_dev,_test}.csv` (offensive language; labels include
`Not_offensive`, `not-Tamil`) and the Malayalam/Kannada equivalents.

### Labels (Tamil sentiment, 5-class)
`Positive`, `Negative`, `Mixed_feelings`, `unknown_state`, `not-Tamil`.
Train split: Positive 19,922 / unknown_state 5,544 / Negative 4,160 / Mixed_feelings 3,907 /
not-Tamil 1,687. (Dev: 2,448 / 729 / 521 / 511 / 189. Test: 2,503 / 631 / 547 / 510 / 211.)

### `DravidianCodeMix` parsing caveat
A few rows contain the separator inside the text, and the column order differs between the "full"
file (label first) and the train/dev/test files (label last). Take the label as the **last** field
of train/dev/test (rstrip a trailing `;` for dev/test), and the **first** field of
`tamil_sentiment_full.csv`.
