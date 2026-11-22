# Preprocessing log - DravidianCodeMix-Tamil

| step | rows | delta |
|---|---:|---:|
| parsed | 44,020 |  |
| after cleaning | 44,020 | +0 |
| drop empty text_clean | 44,020 | +0 |
| drop invalid/missing label | 44,020 | +0 |

## Signal audit (rows containing, before cleaning)

| signal | rows |
|---|---:|
| has_!?_punct | 3,250 |
| has_digit | 7,141 |
| has_emoji | 2,016 |
| has_hashtag | 1,195 |
| has_mention | 95 |
| has_url | 12 |

Emojis, punctuation, numbers and elongations are kept in `text_clean`.

## Length stats (text_clean: min / median / max)

| rows | tokens | chars |
|---:|---|---|
| 44,020 | 1 / 8 / 127 | 2 / 56 / 1124 |

## Labels

| label | rows |
|---|---:|
| Mixed_feelings | 4,928 |
| Negative | 5,228 |
| Positive | 24,873 |
| not-Tamil | 2,087 |
| unknown_state | 6,904 |

## Normalization (Phase E)

- PLD dictionary entries: 9,388
- PLD applied to this dataset: no (dictionary is Telugu-specific)
- Rows changed by normalization: 0 / 44,020
- Monolingual rows detected: 0
- Monolingual rows dropped: no (kept by default)

De-duplication and splitting are intentionally NOT done here.
