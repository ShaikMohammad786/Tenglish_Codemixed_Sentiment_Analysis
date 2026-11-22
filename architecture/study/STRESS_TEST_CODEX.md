# Independent Stress Test: OR-MuRIL

**VERDICT: REFRAME** — keep the orthographic-robustness question; remove the false MuRIL premise and do not present augmentation plus JS loss as a novel architecture.

## 1. Premise: the “Indic Pretraining Paradox” (Plan §§2.1, 3; Roadmap §3.1)

The premise as written is false. MuRIL uses transliterated native-to-Latin pairs in TLM pretraining, including Dakshina Telugu and transliterated Wikipedia, and is evaluated on transliterated text. It outperforms mBERT there. [MuRIL, §§2–3](https://arxiv.org/abs/2103.10730) “Cannot activate Indic subwords” and “forced to use generic English WordPiece” are misleading: its vocabulary includes Latin pieces. RQ1 survives in narrower form: **does standardized transliteration exposure transfer to spontaneous, noisy, code-mixed Roman Telugu, and when does tokenization predict errors?** Measure this; the .8293 LogReg vs .7863 CNN+BiLSTM gap says nothing causal about MuRIL or tokenization.

## 2. Novelty: OR-MuRIL (Plan §§3–4.2)

Two-view CE plus prediction consistency under perturbations is established consistency training, not a new architecture. Closest art: UDA clean/noised consistency (including supervised-only variants), R-Drop bidirectional KL across dropout passes, and VAT local adversarial consistency. [UDA](https://aclanthology.org/2021.emnlp-main.408/), [R-Drop](https://papers.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html), [VAT](https://arxiv.org/abs/1704.03976). JS instead of KL is insufficient novelty. §4.1’s hard-coded mappings are not “data-driven” and may alter meaning. A careful low-resource study could fit DravidianLangTech; TALLIP needs broader evidence. Study **naturally attested, speaker-validated variants** and test on a held-out perturbation family; compare against augmentation+CE.

## 3. Feasibility and compute (Plan §§5–7)

Table 1’s 3 seeds/configuration means **6 trainings**, not the “exactly two runs” in §5. At 27,670 rows, batch 32, three epochs, each takes ~2,595 updates; OR-MuRIL adds a second view. The 25+35 min estimate may fit **one seed pair** with dynamic padding and ideal conditions, not all six. Budget **2–4 T4 GPU-hours** for the six trainings, plus validation/inference; profile first. Batch 32 at length 128 may need fp16, dynamic padding or gradient accumulation. CMTET/stress-test inference is not necessarily CPU-cheap.

## 4. Validity: perturbation test (Plan §6, Table 4)

Not strictly tautological if test comments are unseen, but it is **in-distribution augmentation evaluation**, not real unseen robustness: training and testing reuse the same edit families. Keep comments/groups disjoint; speaker-check label preservation; train on some edit families and reserve another plus natural variants for test. Report clean/perturbed F1, prediction consistency and confidence intervals.

## 5. Controls and CMTET validity (Plan §§5–6)

Essential control: **MuRIL trained on identical augmented views with CE only**; otherwise JS’s effect is unknown. Also report existing char+word TF-IDF LogReg (.8293), character-only, and vanilla MuRIL. ByT5/XLM-R are useful if compute allows; ensembling is optional. CMTET is a *transfer probe*, not clean external validation: domain and label-policy shifts are confounded; 55.36% of multi-row near-duplicate clusters conflict in label, with a spam cluster of 61. Report cluster/source-stratified metrics and label-conflict sensitivity; don’t call the whole set gold data.

## 6. Bigger picture and pivot (Plan §§1–3; companions)

The strongest signal is the linear model beating CNN+BiLSTM by 4.3 F1 points. OR-MuRIL asserts a tokenizer cause without any MuRIL baseline. Leakage is hygiene, not a headline; MDS-Net’s script claim lacks paired script labels. Return to SCRIPT-D/DravidianBench-X as a **benchmark** direction, not a claim of a new network; omit native-script tests until real pairs exist and separate domain, spelling, and label reliability.

## Better RQs / what I would do instead

- **Error mechanism:** After controlling for word frequency, English proportion, length, and class, does MuRIL fertility predict errors on spontaneous Roman Telugu?
- **Robustness:** Do models survive naturally observed, human-validated spelling variants that were never used in training?
- **Transfer:** How much NITANP→CMTET degradation remains after separating domain shift from duplicate clusters and conflicting labels?
- **Benchmark:** Can a compact, reproducible Telugu-English suite report clean, spelling-shift, and cross-corpus performance with confidence intervals?

**Single best accuracy lever:** tune the already strongest char+word TF-IDF baseline on validation only (n-gram ranges, analyzer, regularization), then use targeted, label-verified variant augmentation if errors support it. Current evidence gives no reason to expect OR-MuRIL to beat it.
