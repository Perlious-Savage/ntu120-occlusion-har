# Theme C: Calibration, Uncertainty, Failure Detection, Selective Prediction, and Temporal Reliability

Stage: ARS `lit-review`, Phase 2 (bibliography + source verification). This is not synthesis. Positioning notes are limited to one "Relation to project M4" line per source.
Project: "Uncertainty-aware occlusion-robust RGB-skeleton action recognition on NTU RGB+D 120" (frozen R3D-18 + CTR-GCN, synthetic corruptions; M4 = entropy uncertainty, temperature scaling, ECE/NLL/Brier, AUROC error detection, uncertainty-guided fusion, reliability head, 8-segment temporal reliability, selective prediction/AURC).

---

## 1. Search Strategy

**Databases / sources**: WebSearch (general web index); publisher and proceedings pages (PMLR, papers.nips.cc / proceedings.neurips.cc, CVF Open Access, ECVA, AAAI OJS, Springer LNCS); arXiv abstract pages and ar5iv full-text HTML; Crossref REST API (DOI metadata); OpenReview API (forum IDs); MLAnthology (BibTeX/pages).
**Last searched**: 2026-11-08.
**Keywords (main)**: "on calibration of modern neural networks"; "Bayesian binning" ECE; Brier 1950; "predictive uncertainty under dataset shift"; "misclassified and out-of-distribution examples" baseline; "selective classification" deep neural networks; AURC "bias-reduced uncertainty estimation"; "failure detection" evaluation practices; "evidential deep learning"; "deep ensembles"; "dropout as a Bayesian approximation"; "post-hoc calibration domain drift"; "parameterized temperature scaling"; "revisiting the calibration"; "learning model confidence" true class probability; "rethinking confidence calibration failure prediction"; "temporal segment networks"; "open set action recognition" evidential; "calibration video action recognition"; "uncertainty-aware skeleton action recognition"; "uncertainty attention frames action"; "salient clips"; "trusted multi-view"; "dynamical fusion trustworthy multimodal"; "dynamic fusion low-quality multimodal"; "calibrating multimodal learning".
**Boolean (as typed into WebSearch)**: exact-title phrase + first-author surname + venue/year; topical queries as (calibration OR uncertainty OR confidence) AND (video OR action OR skeleton) AND (shift OR corruption OR open set OR frame OR clip).
**Date range**: 1950 to 2026; seminal works kept regardless of age; topical video/skeleton work 2019 to 2026.
**Inclusion criteria**: (a) peer-reviewed conference/journal paper (or clearly flagged preprint when it fills a direct gap); (b) addresses calibration metrics, post-hoc calibration, uncertainty estimation, failure/error detection, selective prediction, uncertainty-weighted multimodal fusion, or per-frame/segment reliability in video; (c) metadata (title, authors, venue, year, DOI or official URL) confirmed against at least one primary record (publisher/proceedings page or Crossref/OpenReview).
**Exclusion criteria**: pure OOD-detection methods without failure-detection evaluation; regression-only uncertainty; sources whose metadata could not be matched (gray zone = FAIL).
**Coverage Distribution Advisory**:

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: methodological distribution
- Concentration: computational/empirical ML (method or benchmark) = 24/25 (96%)
- Advisory: Expected for this RQ; the one exception is Brier (1950), a methodological note. No qualitative or theoretical-statistics work on proper scoring rules beyond Brier is included.
- Search response: no expansion; scope set by the caller.

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: venue tier distribution
- Concentration: top ML/CV conferences (NeurIPS/ICML/ICLR/CVPR/ICCV/ECCV/AAAI) = 22/25 (88%)
- Advisory: Coverage signal only. Journal venues are underrepresented; one arXiv-only 2026 preprint is included and flagged.
- Search response: no expansion; conference venues are the primary outlets in this field.

Time distribution: no single bucket reaches 70% (2015 and earlier: 3; 2016-2019: 10; 2020-2023: 11; 2024+: 1).

### PRISMA-style flow

```
Records identified (total): ~260 search hits
|-- WebSearch result lists (35 queries x ~10 hits): ~255 (incl. mirrors)
+-- Crossref / OpenReview lookups (direct metadata): 21 DOIs, 5 title queries
Duplicates / mirrors removed (arXiv vs proceedings, aggregator pages): ~205
Unique records screened (title/abstract): 54
Records excluded at title/abstract: 25 (off-theme: pure OOD detectors, domain-specific
    driver monitoring, zero-shot TAD, 2025-26 journal papers in unindexed venues,
    efficiency-only clip selection variants)
Full-text / primary-record assessed: 29
Excluded after assessment (verified but out of scope or redundant): 4
    - Hendrycks & Dietterich (2019) ImageNet-C: corruption benchmark (belongs to robustness theme)
    - Kumar, Liang & Ma (2019) Verified Uncertainty Calibration: ECE-estimator bias; claim not confirmed beyond abstract
    - Zhao et al. (2023) MULE, CVPR: multi-actor open-set; redundant with DEAR
    - Wang & Koniusz (2022) Uncertainty-DTW, ECCV: few-shot skeleton alignment; tangential
Sources included: 25 (24 peer-reviewed, 1 preprint)
UNVERIFIED (not annotated): 0 primary targets; see Section 4 for partially confirmed details
```

---

## 2. Source Verification Report

**Sources reviewed**: 29 | **Verified (included)**: 25 | **Flagged**: 2 (preprint; page-number discrepancies) | **Rejected**: 0 fabricated.

Verification tiers: `VERIFIED` = DOI resolved via Crossref with matching title/authors, or official proceedings page (PMLR / NeurIPS / CVF / ECVA / AAAI) fetched or returned with matching metadata. `PLAUSIBLE` = arXiv + conference listing only, no DOI. Semantic Scholar API was not queried (`[S2-API-UNAVAILABLE]`: not attempted in this run); Crossref and OpenReview were used instead.

| # | Source | Level* | Venue | Author | Method | Currency | COI | Verification |
|---|---|---|---|---|---|---|---|---|
| 1 | Brier 1950 | Foundational | pass | pass | n/a | seminal | pass | VERIFIED (Crossref DOI) |
| 2 | Naeini et al. 2015 | III | pass | pass | pass | seminal | pass | VERIFIED (Crossref DOI) |
| 3 | Guo et al. 2017 | III | pass | pass | pass | seminal | pass | VERIFIED (PMLR page) |
| 4 | Minderer et al. 2021 | III | pass | pass | pass | pass | warn (Google authors, Google models; intellectual) | VERIFIED (NeurIPS PDF + poster page) |
| 5 | Gal & Ghahramani 2016 | III | pass | pass | pass | seminal | pass | VERIFIED (PMLR page) |
| 6 | Lakshminarayanan et al. 2017 | III | pass | pass | pass | seminal | pass | VERIFIED (NeurIPS proceedings + metadata JSON) |
| 7 | Sensoy et al. 2018 | III | pass | pass | warn (reviewers questioned novelty) | pass | pass | VERIFIED (NeurIPS proceedings) |
| 8 | Ovadia et al. 2019 | III (benchmark) | pass | pass | pass | pass | warn (ensembles co-author Lakshminarayanan; intellectual) | VERIFIED (NeurIPS proceedings) |
| 9 | Tomani et al. 2021 | III | pass | pass | pass | pass | pass | VERIFIED (Crossref DOI) |
| 10 | Tomani et al. 2022 | III | pass | pass | pass | pass | pass | VERIFIED (Crossref DOI) |
| 11 | Hendrycks & Gimpel 2017 | III | pass | pass | pass | seminal | pass | VERIFIED (OpenReview forum Hkg4TI9xl) |
| 12 | Geifman & El-Yaniv 2017 | III | pass | pass | pass | seminal | pass | VERIFIED (NeurIPS PDF) |
| 13 | Geifman et al. 2019 | III | pass | pass | pass | pass | pass | VERIFIED (OpenReview forum SJfb5jCqKm) |
| 14 | Corbière et al. 2019 | III | pass | pass | pass | pass | pass | VERIFIED (NeurIPS proceedings) |
| 15 | Jaeger et al. 2023 | III (benchmark) | pass | pass | pass | pass | pass | VERIFIED (OpenReview forum YnkGMIh0gvX) |
| 16 | Zhu et al. 2022 | III | pass | pass | pass | pass | pass | VERIFIED (Crossref DOI) |
| 17 | Han et al. 2021 (TMC) | III | pass | pass | pass | pass | pass | PLAUSIBLE+ (arXiv + ICLR poster page + OpenReview PDF; no DOI) |
| 18 | Han et al. 2022 (MM Dynamics) | III | pass | pass | pass | pass | pass | VERIFIED (Crossref DOI) |
| 19 | Zhang et al. 2023 (QMF) | III | pass | pass | pass | pass | pass | VERIFIED (PMLR page) |
| 20 | Ma et al. 2023 (CML) | III | pass | pass | pass | pass | pass | VERIFIED (PMLR page) |
| 21 | Wang et al. 2016 (TSN) | III | pass | pass | pass | seminal | pass | VERIFIED (Crossref DOI) |
| 22 | Roitberg et al. 2021 (CARING) | III | pass | pass | pass | pass | pass | VERIFIED (Crossref DOI) |
| 23 | Bao et al. 2021 (DEAR) | III | pass | pass | warn (repo notes a corrected HSIC bug) | pass | pass | VERIFIED (Crossref DOI) |
| 24 | Guo, H. et al. 2022 | III | pass | pass | pass | pass | pass | VERIFIED (Crossref DOI) |
| 25 | Korbar et al. 2019 (SCSampler) | III | pass | pass | pass | pass | warn (Facebook AI; industrial lab) | VERIFIED (Crossref DOI) |
| 26 | Khanal & Zhou 2026 | VI (preprint) | warn (arXiv only, not peer-reviewed) | pass | not assessed | pass | pass | PLAUSIBLE (arXiv abstract page) |

*Level: the ARS I-VII hierarchy is clinical; for ML papers, "III" = controlled computational experiment with baselines, "VI" = single unreviewed study. Entry 26 is counted inside the 25 below only as a flagged supplement (see Section 3.6); the core count is 25 peer-reviewed + 1 preprint.

**Flagged sources (detail)**
- **Khanal & Zhou (2026)**. Issue: arXiv-only, 6 pages, no venue. Severity: Medium. Recommendation: include with caveat ("preprint") and do not lean on it as sole support. Evidence: arXiv:2603.15574 comments field lists no venue.
- **Bao et al. (2021) page numbers**. Issue: CVF Open Access lists pp. 13349-13358; Crossref/IEEE DOI lists pp. 13329-13338. Severity: Low. Recommendation: cite the DOI pages.
- **Han et al. (2022) / Tomani et al. (2021) page numbers**. Same CVF vs IEEE pagination mismatch (CVF 20707-20717 vs IEEE 20675-20685; CVF 10124-10132 vs IEEE 10119-10127). Cite DOI pages.

**Predatory journal alerts**: none (all venues are established conferences or AMS journal).
**COI disclosures**: no financial COIs identified; intellectual COIs flagged in matrix (authors evaluating their own method family).
**Verification limitations**: Semantic Scholar not queried; full-text reading was limited to the sections listed under each entry; NeurIPS 2021 page numbers for Minderer et al. not confirmed and therefore omitted.

---

## 3. Annotated Bibliography (N = 25 peer-reviewed + 1 flagged preprint)

Format per entry: APA 7 citation + link; read scope; WHY / HOW / WHAT; Method weaknesses (per ARS rules; `not assessed` when only abstract/metadata read); Relation to project M4.

### 3.1 Calibration metrics and post-hoc calibration

**1. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1-3.** https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
- Read scope: metadata (Crossref) + secondary descriptions.
- WHY: Forecast verification needed a single score rewarding both sharpness and correctness of probabilistic forecasts.
- HOW: Defines the mean squared difference between forecast probability vector and the one-hot outcome.
- WHAT: The Brier score, a strictly proper scoring rule. Note: the original multi-category form sums over classes, so it is twice the common binary Brier score; state which form is reported.
- Method weaknesses: not assessed (read scope: metadata).
- Relation to project M4: Grounds the Brier metric. Report the multi-class form (sum over 120 classes, mean over samples) and say so explicitly.

**2. Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence, 29*(1), 2901-2907.** https://doi.org/10.1609/aaai.v29i1.9602
- Read scope: abstract + metadata.
- WHY: Classifier scores are often miscalibrated; existing binning calibrators depend on a single bin choice.
- HOW: Bayesian Binning into Quantiles (BBQ): Bayesian averaging over equal-frequency binning models, scored by a BDeu-style marginal likelihood.
- WHAT: BBQ calibrator; the paper is the standard citation for the binned Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) definitions.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Cite for the ECE/MCE definition; fix and report the bin count (e.g., 15 equal-width) since ECE depends on binning.

**3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR Vol. 70, pp. 1321-1330).** https://proceedings.mlr.press/v70/guo17a.html
- Read scope: abstract + proceedings metadata.
- WHY: Modern deep CNNs are more accurate but less calibrated than older networks.
- HOW: Studies effects of depth, width, weight decay, BatchNorm on calibration; compares post-hoc calibrators on a held-out validation set using ECE, NLL, reliability diagrams.
- WHAT: Single-parameter temperature scaling is surprisingly effective in-distribution and leaves accuracy unchanged.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Directly grounds "temperature scaling fit on validation only" per branch. Its evidence is i.i.d. only, so it does not license applying a clean-val temperature to corrupted inputs (see entries 8, 9).

**4. Minderer, M., Djolonga, J., Romijnders, R., Hubis, F., Zhai, X., Houlsby, N., Tran, D., & Lucic, M. (2021). Revisiting the calibration of modern neural networks. In *Advances in Neural Information Processing Systems* (Vol. 34).** https://papers.neurips.cc/paper_files/paper/2021/file/8420d359404024567b5aefda1231af24-Paper.pdf (arXiv:2106.07998)
- Read scope: abstract.
- WHY: Revisit Guo et al.'s conclusions for recent architectures (ViT, MLP-Mixer, big pretraining).
- HOW: Large-scale comparison of in-distribution and shifted-distribution calibration across model families.
- WHAT: Recent non-convolutional models are among the best calibrated; decay of calibration with distribution shift is less pronounced in recent architectures; architecture is a major determinant of calibration.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Architecture matters: expect R3D-18 (3D CNN) and CTR-GCN (GCN) to miscalibrate differently, so fit and report temperature per branch, not jointly.

### 3.2 Uncertainty estimation methods (baselines/alternatives to softmax entropy)

**5. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In *Proceedings of the 33rd International Conference on Machine Learning* (PMLR Vol. 48, pp. 1050-1059).** https://proceedings.mlr.press/v48/gal16.html
- Read scope: abstract.
- WHY: Deep nets lack model (epistemic) uncertainty without expensive Bayesian inference.
- HOW: Casts dropout training as approximate variational inference in deep Gaussian processes; MC dropout at test time.
- WHAT: Cheap uncertainty estimates from existing dropout networks.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Possible ablation baseline for entropy; with frozen backbones, MC dropout would apply only to heads, which Ovadia et al. (entry 8) report as weaker than full dropout under shift.

**6. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 6402-6413).** https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
- Read scope: abstract + proceedings metadata.
- WHY: Bayesian NNs are costly and hard to tune; need a scalable alternative.
- HOW: Train several independently initialised networks (proper scoring rule loss, optional adversarial training) and average predictions.
- WHAT: Deep ensembles give well-calibrated uncertainty and higher uncertainty on unseen-class data.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Strongest known baseline under shift (entry 8), but infeasible for retraining frozen R3D-18/CTR-GCN; cite to justify scoping (head-level ensembles at most) and as a stated limitation.

**7. Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. In *Advances in Neural Information Processing Systems* (Vol. 31, pp. 3179-3189).** https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html
- Read scope: abstract + public reviews page (summary).
- WHY: Softmax gives point estimates and cannot express "I don't know"; Bayesian weight posteriors are costly.
- HOW: Network outputs non-negative evidence parameterising a Dirichlet over class probabilities (subjective logic); uncertainty mass u = K/S.
- WHAT: Deterministic single-pass uncertainty; reported strong OOD detection and adversarial robustness.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Alternative to entropy for per-branch and per-segment uncertainty; it underlies DEAR (entry 23) and TMC fusion (entry 17). Would require retraining a head, consistent with the M4 reliability head plan.

### 3.3 Calibration under dataset shift (the reviewers' concern)

**8. Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J., Lakshminarayanan, B., & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In *Advances in Neural Information Processing Systems* (Vol. 32, pp. 13991-14002).** https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html
- Read scope: sections (abstract; Sections 3-5; appendix metric list) via ar5iv.
- WHY: Uncertainty methods were evaluated almost only in-distribution, yet deployment involves shift.
- HOW: Large benchmark (images incl. corrupted ImageNet/CIFAR, text, ad-click) comparing vanilla, temperature scaling, MC dropout, last-layer methods, SVI, deep ensembles; metrics: accuracy, ECE, NLL, Brier (with reliability/resolution decomposition), entropy, accuracy-vs-confidence curves.
- WHAT: Temperature scaling fit on the i.i.d. validation set is well calibrated on i.i.d. test and small shift but "significantly outperformed by methods that take epistemic uncertainty into account as the shift increases"; ECE rises markedly with shift intensity; deep ensembles (about 5 members) most robust.
- Method weaknesses (empirical benchmark; read: Sections 3-5, appendix)
  - checked: found: shift-intensity sweep, multiple metrics, multiple data modalities | not checked: hyperparameter sensitivity per method, video data.
  - Calibration is fit on a clean i.i.d. validation set for all post-hoc methods; results would differ if any validation set contained shifted data. [reader-inferred]
- Relation to project M4: Primary evidence for the reviewers' concern. It predicts that a clean-val temperature will under-correct overconfidence as corruption severity rises. M4 should report ECE/NLL/Brier per corruption type and severity, not only clean.

**9. Tomani, C., Gruber, S., Erdem, M. E., Cremers, D., & Buettner, F. (2021). Post-hoc uncertainty calibration for domain drift scenarios. In *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 10119-10127).** https://doi.org/10.1109/CVPR46437.2021.00999
- Read scope: sections (method, results) via ar5iv.
- WHY: Post-hoc calibrators are tuned in-domain and become overconfident under domain drift.
- HOW: Fit calibrators (TS, ETS, IR, IRM, TS-IR) on a validation set perturbed with Gaussian noise at 10 tuned strengths so accuracy decays roughly linearly (the "-P" variants).
- WHAT: Perturbed-validation calibration substantially improves calibration under shift; TS-P/ETS-P pay a clear in-domain calibration cost, IR-P does not.
- Method weaknesses (empirical; read: method + results sections)
  - checked: found: multiple architectures/datasets, in-domain cost reported | not checked: real (non-synthetic) shift beyond the paper's benchmarks.
  - Perturbation family is generic Gaussian noise; would mislead when test-time shift is structurally different (e.g., joint dropout or occlusion masks). [reader-inferred]
- Relation to project M4: Gives the direct fix for the reviewers' concern: a "corruption-aware calibration" ablation fitting temperature on a corrupted validation split (held-out corruption types to avoid leakage), compared with clean-val TS, reporting both clean and corrupted ECE.

**10. Tomani, C., Cremers, D., & Buettner, F. (2022). Parameterized temperature scaling for boosting the expressive power in post-hoc uncertainty calibration. In S. Avidan et al. (Eds.), *Computer Vision - ECCV 2022* (LNCS Vol. 13673, pp. 555-569). Springer.** https://doi.org/10.1007/978-3-031-19778-9_32
- Read scope: abstract + metadata.
- WHY: Accuracy-preserving post-hoc calibrators (single temperature) have limited expressive power.
- HOW: A small network predicts a per-sample temperature from the logits (PTS), keeping argmax unchanged.
- WHAT: PTS outperforms existing post-hoc calibrators across architectures, datasets and metrics (authors' experiments).
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Middle ground between one global temperature and a learned reliability head; per-sample temperature could absorb corruption-dependent overconfidence without changing predictions.

### 3.4 Failure detection and selective prediction

**11. Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. In *5th International Conference on Learning Representations (ICLR 2017)*.** https://openreview.net/forum?id=Hkg4TI9xl (arXiv:1610.02136)
- Read scope: abstract + metadata.
- WHY: No standard baseline for detecting errors or OOD inputs from a trained classifier.
- HOW: Use the maximum softmax probability (MSP); evaluate error and OOD detection with AUROC and AUPR across vision, NLP and speech.
- WHAT: Correct predictions tend to have higher MSP than errors or OOD inputs; MSP is a strong but beatable baseline.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Grounds the "AUROC of uncertainty for detecting branch errors" protocol and MSP as the baseline score that entropy must be compared against.

**12. Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4878-4887).** https://papers.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf
- Read scope: abstract + metadata.
- WHY: Mission-critical use needs a classifier that can abstain to guarantee a target risk.
- HOW: Threshold a confidence rate function (softmax response or MC dropout) on a trained net; the threshold is chosen with a risk guarantee (selection with guaranteed risk).
- WHAT: Risk-coverage framework for deep nets; e.g., 2% top-5 ImageNet error at 99.9% probability with almost 60% coverage.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Grounds the selective-prediction setup and risk-coverage curves; SR (softmax response = MSP) was competitive with MC dropout here.

**13. Geifman, Y., Uziel, G., & El-Yaniv, R. (2019). Bias-reduced uncertainty estimation for deep neural classifiers. In *7th International Conference on Learning Representations (ICLR 2019)*.** https://openreview.net/forum?id=SJfb5jCqKm (arXiv:1805.08206)
- Read scope: sections (Section 3 definitions; Section 7 setup) via ar5iv.
- WHY: Confidence estimates from the final SGD-trained model are biased for highly confident points.
- HOW: Pointwise/Averaged Early Stopping using earlier training snapshots; evaluates confidence scores (softmax response, NN-distance, MC dropout, ensemble) by selective classification.
- WHAT: Defines AURC (area under the empirical risk-coverage curve) and E-AURC = AURC - AURC of the optimal ranking ("unitless, in [0,1]") in Section 3.
- Method weaknesses (empirical; read: Sections 3, 7)
  - checked: found: multiple confidence functions, multiple datasets | not checked: robustness of snapshot method under shift.
  - AURC mixes base accuracy and ranking quality; comparisons across models with different accuracy are confounded unless E-AURC is also reported. [author-acknowledged, section: 3 (motivation for E-AURC)]
- Relation to project M4: Cite for AURC/E-AURC definitions; report both, since corrupted settings change base accuracy.

**14. Corbière, C., Thome, N., Bar-Hen, A., Cord, M., & Pérez, P. (2019). Addressing failure prediction by learning model confidence. In *Advances in Neural Information Processing Systems* (Vol. 32, pp. 2902-2913).** https://papers.nips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html
- Read scope: sections (method, metrics) via ar5iv.
- WHY: MCP "leads by design to high confidence values, even for erroneous ones", so correct/incorrect MCP distributions overlap.
- HOW: ConfidNet: an auxiliary head on the frozen classifier's intermediate features regresses the True Class Probability (TCP = p(y_true|x)) with MSE (beat BCE, focal and ranking losses), then optional encoder fine-tuning.
- WHAT: Better failure prediction (AUPR-Error primary; also FPR@95TPR, AUROC, risk-coverage curves) than MCP, MC dropout, trust score.
- Method weaknesses (empirical; read: method + metrics sections)
  - checked: found: ablation of losses, multiple datasets | not checked: behaviour under covariate shift.
  - TCP targets are learned on the training set where the classifier is mostly correct, so failures are scarce; the authors report training on validation performed worse. Distorts results when train-set TCP differs from test-time TCP (e.g., under corruption). [author-acknowledged, section: training details] + [reader-inferred for the shift part]
- Relation to project M4: Directly grounds the "reliability head regressing p(y_true)" design, including frozen backbone + feature-input head + MSE. Warns: training targets from clean train data will be over-optimistic; corruption-aware training of the head is the natural remedy.

**15. Jaeger, P. F., Lüth, C. T., Klein, L., & Bungert, T. J. (2023). A call to reflect on evaluation practices for failure detection in image classification. In *11th International Conference on Learning Representations (ICLR 2023)*.** https://openreview.net/forum?id=YnkGMIh0gvX (arXiv:2211.15259)
- Read scope: sections (Sections 2-4, Table 2, Appendix C summaries) via ar5iv, first ~68% of text.
- WHY: Misclassification detection, OOD detection, selective classification and calibration are evaluated with separate, inconsistent protocols although they share the goal of detecting failures.
- HOW: Defines requirements for failure-detection evaluation; benchmarks many confidence scoring functions on i.i.d., 15 corruption types x 5 severities (CIFAR-10/100-C), sub-class shifts (iWildCam, BREEDS, CAMELYON), and new-class shifts.
- WHAT: Proposes AURC as primary metric (penalises CSFs that hurt accuracy; AUROC_f suffers from shifting failure labels). Softmax baselines (MSR, MCD-MSR) are best or near-best on i.i.d. and all covariate shifts except new-class shifts; no general advantage of entropy over MCD-MSR; calibration and ranking are separate requirements; temperature scaling/64-bit logits recommended to avoid softmax rounding errors that damage ranking.
- Method weaknesses (empirical benchmark; read: Sections 2-4, part of appendices)
  - checked: found: shift taxonomy, multiple architectures (CNN, ViT) | checked: none found: deep ensembles (not evaluated) | not checked: final ~32% of appendix.
  - Image classification only; corruptions are image-level, so conclusions may not transfer to skeleton joint corruption or temporal occlusion. [reader-inferred]
- Relation to project M4: Most important positioning source. It supports MSP/entropy as strong failure detectors under corruption shift (counter to the reviewers' concern), but requires AURC as headline metric, separate reporting of calibration (ECE) and ranking (AUROC/AURC), and comparing entropy against MSP rather than assuming entropy is better.

**16. Zhu, F., Cheng, Z., Zhang, X.-Y., & Liu, C.-L. (2022). Rethinking confidence calibration for failure prediction. In S. Avidan et al. (Eds.), *Computer Vision - ECCV 2022* (LNCS Vol. 13685, pp. 518-536). Springer.** https://doi.org/10.1007/978-3-031-19806-9_30
- Read scope: abstract + secondary summary.
- WHY: Calibration methods are assumed to also help decide when to trust a prediction.
- HOW: Evaluates training-time and post-hoc calibration methods on failure prediction (AURC, AUROC, FPR95); links flat minima to confidence separation.
- WHAT: Most calibration methods are useless or harmful for failure prediction (reduced separability of correct vs. incorrect); post-hoc TS does not improve it; flat-minima training (FMFP) does.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Warns that good ECE after temperature scaling says nothing about error-detection AUROC/AURC (temperature scaling leaves predictions unchanged and typically reorders cross-sample MSP/entropy only slightly; reader-inferred). M4 should not present TS as improving failure detection.

### 3.5 Uncertainty-guided multimodal fusion

**17. Han, Z., Zhang, C., Fu, H., & Zhou, J. T. (2021). Trusted multi-view classification. In *9th International Conference on Learning Representations (ICLR 2021)*.** https://arxiv.org/abs/2102.02051 (ICLR page: https://iclr.cc/virtual/2021/poster/2908)
- Read scope: abstract.
- WHY: Multi-view fusion should judge, per sample, how much each view can be trusted.
- HOW: Each view outputs evidence for a Dirichlet; views combined with Dempster-Shafer rule; per-view uncertainty mass down-weights unreliable views.
- WHAT: Gains in accuracy, reliability and robustness (noisy views). Extended in IEEE TPAMI 2023 (doi:10.1109/TPAMI.2022.3171983).
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Principled template for uncertainty-guided RGB/skeleton fusion; direct alternative to entropy-weighted averaging.

**18. Han, Z., Yang, F., Huang, J., Zhang, C., & Yao, J. (2022). Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 20675-20685).** https://doi.org/10.1109/CVPR52688.2022.02005
- Read scope: abstract + secondary summary.
- WHY: Static fusion is not trustworthy when modality informativeness varies by sample (safety-critical, medical).
- HOW: Sparse gating for feature-level informativeness; TCP-based confidence per modality used to weight modality-level fusion.
- WHAT: Superior accuracy and trustworthiness on multimodal medical classification.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Closest published analogue of "reliability head regressing p(y_true) per branch, used as fusion weight". Must be cited as prior art for that component.

**19. Zhang, Q., Wu, H., Zhang, C., Hu, Q., Fu, H., Zhou, J. T., & Peng, X. (2023). Provable dynamic fusion for low-quality multimodal data. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 41753-41769).** https://proceedings.mlr.press/v202/zhang23ar.html
- Read scope: sections (method Eq. 9, Eqs. 14-16, experiment setup) via arXiv HTML.
- WHY: Dynamic fusion lacks theory explaining when it beats static fusion on low-quality data.
- HOW: QMF: modality weight is a decreasing linear function of energy-based uncertainty from each unimodal classifier's logits; a ranking regulariser aligns weights with per-sample historical training loss. Tested with 50% of modalities corrupted by Gaussian (image) or blank (text) noise.
- WHAT: Generalisation bound showing uncertainty-aware dynamic fusion can beat static fusion; improved accuracy and robustness.
- Method weaknesses (empirical + theoretical; read: method and setup)
  - checked: found: corruption protocol, ablation | not checked: proof assumptions.
  - Corruption applied during evaluation is synthetic and modality-wide; does not test partial/temporal corruption. [reader-inferred]
- Relation to project M4: Strong baseline for uncertainty-guided fusion; suggests energy score as a third uncertainty measure next to entropy and MSP; its theory supports the M4 claim that dynamic weighting helps under corruption.

**20. Ma, H., Zhang, Q., Zhang, C., Wu, B., Fu, H., Zhou, J. T., & Hu, Q. (2023). Calibrating multimodal learning. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 23429-23450).** https://proceedings.mlr.press/v202/ma23i.html
- Read scope: abstract.
- WHY: Multimodal classifiers' confidence can rise when a modality is corrupted or removed.
- HOW: CML regulariser enforcing that confidence should not increase when a modality is removed.
- WHAT: Improved calibration, accuracy and robustness across multimodal models.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Provides a concrete diagnostic for M4 error analysis: check whether fused confidence rises when RGB or skeleton is corrupted; motivates a monotonicity check on the fusion head.

### 3.6 Video and skeleton action recognition: calibration, uncertainty, temporal reliability

**21. Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X., & Van Gool, L. (2016). Temporal segment networks: Towards good practices for deep action recognition. In B. Leibe et al. (Eds.), *Computer Vision - ECCV 2016* (LNCS Vol. 9912, pp. 20-36). Springer.** https://doi.org/10.1007/978-3-319-46484-8_2
- Read scope: abstract + metadata.
- WHY: Dense sampling of long videos is costly and models see only short snippets.
- HOW: Split video into K equal segments, sample one snippet each, aggregate snippet scores with a segmental consensus (e.g., average) under video-level supervision.
- WHAT: State-of-the-art HMDB51 (69.4%) and UCF101 (94.2%) at the time.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Grounds the 8-segment design; TSN's consensus is uniform averaging, so M4's per-segment reliability weighting is a direct, citable departure from TSN.

**22. Roitberg, A., Haurilet, M., Martinez, M., & Stiefelhagen, R. (2021). Uncertainty-sensitive activity recognition: A reliability benchmark and the CARING models. In *2020 25th International Conference on Pattern Recognition (ICPR)* (pp. 3814-3821). IEEE.** https://doi.org/10.1109/ICPR48806.2021.9412540
- Read scope: sections (Section III-D, results tables) via ar5iv.
- WHY: Activity recognition models' confidence values were not evaluated for reliability.
- HOW: ECE/reliability-diagram benchmark on Drive&Act and HMDB-51 with I3D and P3D; CARING = 2-layer network on intermediate backbone features predicting a per-input temperature (accuracy unchanged).
- WHAT: Uncalibrated I3D ECE 20.11% on HMDB-51; temperature scaling cuts it to 7.75%; CARING to 5.98%. Similar pattern on Drive&Act.
- Method weaknesses (empirical; read: III-D and results)
  - checked: found: two backbones, two datasets | checked: none found: evaluation under shift/corruption (checked results section).
  - In-distribution evaluation only; a learned input-dependent temperature may not generalise to corruptions unseen during calibration fitting. [reader-inferred]
- Relation to project M4: Closest video-action calibration precedent: confirms 3D-CNN action models are overconfident and that TS helps in-distribution. Leaves corruption-shift calibration of RGB and skeleton branches open (M4 gap).

**23. Bao, W., Yu, Q., & Kong, Y. (2021). Evidential deep learning for open set action recognition. In *2021 IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 13329-13338).** https://doi.org/10.1109/ICCV48922.2021.01310
- Read scope: sections (EUC loss, evaluation setup) via ar5iv.
- WHY: Open-set action recognition needs uncertainty that separates unknown actions and is not dominated by static scene bias.
- HOW: DEAR: EDL head (Dirichlet, u = K/S) + Evidential Uncertainty Calibration (EUC) loss pushing correct predictions to low uncertainty and wrong ones to high (accurate-certain / inaccurate-uncertain) + contrastive debiasing module; plug-in for I3D, TSM, SlowFast, TPN.
- WHAT: Gains in open-set AUC and open macro-F1 with UCF-101 known and HMDB-51 / MiT-v2 unknown.
- Method weaknesses (empirical; read: EUC + setup)
  - checked: found: multiple backbones, two unknown sets | not checked: closed-set covariate shift.
  - Open-set protocol (new classes) differs from covariate corruption; official repo notes a corrected HSIC implementation with "performance not guaranteed". [author-acknowledged, quote: repository README]
- Relation to project M4: The EUC loss is a ready-made training signal for a reliability head ("errors should be uncertain"), closely related to regressing exp(-CE). Cite as the main video-specific uncertainty precedent; distinguish M4 (corruption shift, multimodal) from DEAR (new classes, RGB only).

**24. Guo, H., Ren, Z., Wu, Y., Hua, G., & Ji, Q. (2022). Uncertainty-based spatial-temporal attention for online action detection. In S. Avidan et al. (Eds.), *Computer Vision - ECCV 2022* (LNCS, pp. 69-86). Springer.** https://doi.org/10.1007/978-3-031-19772-7_5
- Read scope: abstract + metadata.
- WHY: Online action detection on streaming video needs to focus on informative frames and regions.
- HOW: Probabilistic extension of baseline models; predictive uncertainty (mutual information) generates attention over spatial regions and frames; inference fuses baseline and probabilistic streams by input uncertainty.
- WHAT: Improvements on THUMOS-14, TVSeries, HDD, with reported better generalisation across views and occlusions.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Closest precedent for per-frame/segment reliability weighting by uncertainty, explicitly evaluated for occlusion; cite as prior art for the per-segment reliability component.

**25. Korbar, B., Tran, D., & Torresani, L. (2019). SCSampler: Sampling salient clips from video for efficient action recognition. In *2019 IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 6231-6241).** https://doi.org/10.1109/ICCV.2019.00633
- Read scope: abstract.
- WHY: Uniformly averaging clip predictions lets uninformative clips dilute the video-level decision.
- HOW: A lightweight sampler scores clips for saliency; only top clips are passed to the expensive classifier.
- WHAT: About 7% accuracy gain and more than 15x compute reduction on Sports-1M.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Supports the premise that non-uniform segment weighting beats TSN-style averaging; differs from M4 in that saliency is learned for efficiency, not reliability under corruption.

**Flagged supplement (preprint, not peer-reviewed)**

**26. Khanal, A., & Zhou, J. (2026). *Severe domain shift in skeleton-based action recognition: A study of uncertainty failure in real-world gym environments* [Preprint]. arXiv.** https://arxiv.org/abs/2603.15574
- Read scope: abstract.
- WHY: Skeleton action uncertainty methods are evaluated only in-distribution.
- HOW: Skeleton Transformer trained on NTU-120 tested zero-shot on Gym2D and UCF101 skeletons; compares uncertainty/OOD scores and selective classification.
- WHAT: Accuracy collapses (63.2% to 1.6%); energy and Mahalanobis scores reach AUROC >= 0.91 for shift detection, yet risk stays 99.6% at 50% coverage; a fine-tuned gate restores calibration/abstention.
- Method weaknesses: not assessed (read scope: abstract_only).
- Relation to project M4: Only (preprint) evidence specific to NTU-120 skeleton models: high detection AUROC does not imply good risk-coverage, supporting AURC over AUROC as the M4 headline. Cite with "preprint" caveat.

---

## 4. UNVERIFIED / partially confirmed items (not annotated)

No requested primary target failed verification. Details that could not be confirmed and were therefore omitted from citations:
- Minderer et al. (2021) NeurIPS page numbers (not shown on any fetched record).
- Han et al. (2021) ICLR OpenReview forum ID (OpenReview API search returned no ICLR 2021 note; arXiv and iclr.cc page used instead).
- Zhu et al. (2022) LNCS volume number 13685 comes from the ECVA PDF path (136850512); Crossref record did not return the volume field.
- Guo, H. et al. (2022) LNCS volume not shown on fetched records; omitted.
- Claim "binned ECE underestimates true calibration error" (Kumar et al., 2019): not confirmed from abstract; source excluded rather than cited for it.
- Specific uncertainty methods that fail in Khanal & Zhou (2026): abstract does not name MSP/entropy/TS; do not cite it for those specifics.

---

## 5. Search Limitations
- Semantic Scholar not queried; deduplication by title + DOI only.
- Full text was read only for entries 8, 9, 13, 14, 15 (partial), 19, 22, 23; all others are abstract/metadata level, so "Method weaknesses" are marked not assessed.
- No dedicated search hit was found for calibration of skeleton GCNs (e.g., CTR-GCN) under joint corruption in a peer-reviewed venue; the only skeleton-shift evidence is a 2026 preprint. This is a coverage gap, not a search failure claim beyond the queries listed.
- Video calibration under synthetic corruption (as opposed to in-distribution or open-set) had no peer-reviewed hit in these searches.
