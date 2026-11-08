# Related work and Milestone 4 positioning

Narrative/integrative literature review (ARS `lit-review` mode), written 2026-11-08 to answer review item
A6 (position against prior work, fix one M4 claim, decide the title scope). Three themed searches ran
over web search, OpenAlex, Crossref, arXiv and the official proceedings pages (CVF, PMLR, NeurIPS,
OpenReview). 70 unique sources were verified; anything that could not be confirmed was excluded. The
annotated bibliographies (search strings, read scope, per-source weaknesses) are in
`docs/literature/`; this file is the synthesis. All gap statements below are bounded by those searches:
"not found" means not found by them, not proven absent.

AI disclosure: the searches, verification and this synthesis were produced with AI research tools
(Claude) and checked against the source pages listed in the references.

## 1. Map of the field

### 1.1 Learned reliability and dynamic late fusion: the estimator M4 plans already exists

| Work | Reliability quantity | Supervision | Level | Per-sample | Per-time | Corruption tests | Calibration / AURC |
|---|---|---|---|---|---|---|---|
| TCP / ConfidNet (Corbière 2019) | p(y_true) | MSE on frozen classifier | confidence only | yes | no | no | AUPR-error |
| TMC (Han 2021, 2023) | Dirichlet evidence | evidential loss | evidence | yes | no | Gaussian noise on views | no |
| MMDynamics (Han 2022) | p(y_true) per modality | MSE, joint training | feature scaling | yes | no | no | no |
| QMF (Zhang 2023) | energy score | ranking regulariser | logit | yes | no | Gaussian/salt noise | no |
| CML (Ma 2023) | confidence must not rise when a modality is removed | regulariser | logit | yes | no | modality removal | no |
| PDF (Cao 2024) | p_true (collaborative) | MSE | logit weight | yes | no | noise | no |
| UDML (Wei 2026) | injected-noise intensity | regression on known noise | logit | yes | no | graded noise | no |
| **M4 (planned)** | p(y_true), entropy | MSE / none | logit + feature | yes | **8 segments** | **6 families x 5 severities, single + dual** | **ECE/NLL/Brier, AUROC, AURC** |

Consequences for M4:

1. **The reliability head is TCP.** A Linear-GELU-Dropout-Linear-Sigmoid head regressing p(y_true)
   from frozen-branch features is ConfidNet (Corbière 2019); using it as a fusion weight is
   MMDynamics (feature level) and PDF (logit level). PDF is the closest single precedent and must be
   a baseline, not the contribution.
2. **Targets A and B are one target here.** Both branches were trained with plain cross-entropy
   (`src/engine.py`), so exp(-CE) = p(y_true). They differ only after temperature scaling
   (exp(-CE_T) = p_T(y_true)), with label smoothing, or with segment-level vs clip-level CE. M4 should
   replace Target B with a quantity that is actually different, e.g. the calibrated p_T(y_true) or the
   known corruption intensity (UDML).
3. **Fitting the head on held-out data is stacking, with known precedent.** Wolpert (1992) fits a
   combiner on held-out predictions; Gradient-Blending (Wang 2020) sets modality weights from
   validation loss. TCP's authors report the opposite trade-off: a small hold-out set left too few
   errors to learn from. M3's gate saturating onto RGB when trained on training features matches the
   greedy-learner / dominant-modality behaviour (Wu 2022; Peng 2022 OGM-GE; Du 2023).
4. **Corruption-aware reliability training is also not new** (UDML trains on injected noise
   intensity and reports that entropy-type estimators miss slight degradation and still weight heavily
   corrupted modalities). What is untested is whether this transfers to *held-out corruption
   families* and to *temporally localised* corruption in video + skeleton.
5. **QMF gives the AUROC analysis a purpose.** QMF proves dynamic fusion beats static fusion when a
   modality's weight is negatively correlated with its loss. M4's AUROC of branch uncertainty against
   branch errors is a direct empirical test of that condition, per corruption family and severity.

### 1.2 RGB + pose fusion for action recognition

Clean NTU-120 X-Sub / X-Set top-1: RGBPoseConv3D (Duan 2022) 95.3 / 96.4; MMNet (Yu 2023)
92.9 / 94.4; pi-ViT (Reilly 2024, RGB at test) 91.9 / 92.9; VPN (Das 2020) 86.3 / 87.8; our fusion
about 89.9 / 92. **Our backbones are deliberately small and frozen; the project does not compete on
clean accuracy** and should say so. None of these methods weights modalities by estimated
reliability, and none is tested under corrupted inputs. Robustness evidence that motivates
reliability weighting:

- AMR (Liu 2024, NTU-60, adversarial): undefended RGB-skeleton fusion is less robust than the
  skeleton branch alone when one modality is attacked; learned modality reweighting helps.
- ActionMAE (Woo 2023, NTU-60/120, RGB/depth/IR): concatenation fusion collapses when a modality is
  removed.
- Nearest RGB+skeleton reliability work: AMFI-Net (Xu 2025, NTU-60, no corruption), MAGNet (Dong
  2026, missing-modality tests only, no uncertainty), Takami and Fukuda (2026, per-frame OpenPose
  confidence, not NTU).
- Guo (2026, high-school journal, UCF-101) runs frozen RGB + skeleton branches under occlusion with
  probability/log-probability fusion, close to our M3 baseline set; it names uncertainty-aware fusion
  as future work. Cite it to pre-empt a novelty objection.

### 1.3 Skeleton robustness to occlusion and noise

Our skeleton corruptions follow the RA-GCN protocol family (Song 2019, 2021: part occlusion, random
joint occlusion p = 0.2-0.6, jitter, on NTU-60 and NTU-120 X-Set), reused by Tanaka (2024) and
Parts-Mamba (Shen 2026). Other strands: trust gates (Liu 2018), predictive encoding for noise (Yoon
2022), adversarial skeletons (Wang 2021), occluded one-shot and self-supervised recognition (Peng 2023;
Chen 2025), low-quality skeletons via distillation (Liu 2024), temporal corruption (FineTec, Shao
2026), and SHARDeg (Malzard 2025, unreviewed preprint, frame-level degradation of NTU-120 only).
**All of this work is skeleton-only.** PoseC3D reports MS-G3D losing about 14 points under keypoint
dropping, consistent with our CTR-GCN results.

**Natural occlusion is in the data and unused.** NTU stores a Kinect tracking state per joint per
frame (0 not tracked, 1 inferred, 2 tracked; Microsoft 2014). No paper was found that uses inferred
joints as a natural-occlusion test split or as a reliability signal.

### 1.4 Corruption benchmarks for video

ImageNet-C (Hendrycks 2019), Mini Kinetics-C / SSV2-C (Yi 2021) and Kinetics-P / UCF101-P (Schiappa
2023) cover noise, blur, weather, digital and temporal corruptions; READ (Yang 2024) grades 15 video +
6 audio corruptions for audio-visual test-time adaptation. **None has actor-box occlusion, none uses
NTU, none corrupts a skeleton stream.** Yi et al. found that training a 3D ResNet-18 with Gaussian
noise reduced robustness to other corruptions, a warning for corruption-aware training (M4 keeps the
backbones frozen and only trains the reliability/fusion head on corruptions).

### 1.5 Calibration, failure detection and selective prediction

- Temperature scaling (Guo 2017) fitted on clean validation data holds under mild shift but ECE grows
  with shift severity (Ovadia 2019). Fitting the calibrator on a perturbed validation set (Tomani 2021;
  parameterised TS, Tomani 2022) is the established fix and becomes a direct M4 ablation.
- For error detection under corruption, the max-softmax baseline (Hendrycks and Gimpel 2017) is best
  or near-best (Jaeger 2023); entropy must be compared against it. Calibration methods do not improve
  failure prediction (Zhu 2022), so ECE and AUROC/AURC are reported separately and an ECE gain is not
  claimed as better error detection.
- AURC and E-AURC: Geifman and El-Yaniv (2017); Geifman et al. (2019).
- Video/skeleton precedents: CARING (Roitberg 2021; I3D on HMDB-51, ECE 20.1% to 7.75% with
  temperature scaling, in-distribution only); DEAR (Bao 2021, evidential, open-set rather than
  corruption); uncertainty-based frame attention for online detection (Guo 2022); SCSampler (Korbar
  2019, salient clip selection). TSN (Wang 2016) averages segments uniformly, so per-segment
  reliability weighting is a direct departure from it. The only NTU-120 calibration-under-shift
  evidence found is a preprint (Khanal and Zhou 2026). **No peer-reviewed study of calibration of
  video or skeleton action models under synthetic corruption was found.**

## 2. What is new, and what is not

Not new (cite, use as baselines): the p(y_true) reliability target and head (TCP, MMDynamics, PDF);
energy/entropy-weighted late fusion (QMF); corruption-intensity-supervised weighting (UDML);
temperature scaling and ECE; AURC; fitting a combiner on held-out data (stacking).

Not found in the literature (bounded by the logged searches):

1. A graded single- and dual-modality corruption benchmark for RGB + skeleton fusion on NTU-120,
   including actor-box occlusion of the RGB stream (Milestone 3).
2. Reliability-weighted RGB + skeleton fusion evaluated with calibration and selective prediction
   under those corruptions.
3. Per-segment (temporal) modality reliability checked against known corrupted segments: the temporal
   frame mask and body-part dropout give ground truth for which segments are degraded, so segment
   reliability can be scored directly, not only through final accuracy.
4. Transfer of a corruption-aware reliability head to held-out corruption families.
5. Kinect "inferred" joints as a natural-occlusion test of the same reliability estimates.

## 3. Proposed Milestone 4 claim (for agreement with the supervisor)

> On NTU RGB+D 120 with frozen RGB and skeleton branches, weighting each modality per sample and per
> temporal segment by a reliability head trained on corrupted held-out data reduces the conditions in
> which fusion falls below the better single modality, and lowers selective-prediction risk (AURC),
> compared with average fusion and clean-trained confidence weighting (TCP/PDF-style), including on
> corruption families held out from training.

It is falsifiable against M3's own numbers: average fusion is below the healthy modality in 15 of 30
X-Sub single-modality conditions and attention fusion in 13 (X-Set figures follow the A2 retrain).

Design changes this implies for M4:

- Baselines: average fusion, MSP-weighted, entropy-weighted, TCP/PDF head trained on clean data,
  QMF-style energy weighting; the contribution is the corruption-trained, per-segment variant.
- Replace Target B (identical to Target A under plain CE) with calibrated p_T(y_true) or the known
  corruption intensity.
- Leave-one-corruption-family-out training and evaluation for the corruption-aware setting.
- Two calibration fits: clean validation and corrupted validation (Tomani 2021), reported per severity.
- Report MSP alongside entropy for AUROC and AURC; report ECE separately from failure detection.
- Score segment reliability against the known masked or dropped segments.

## 4. Title scope (decision for the supervisor)

RGB actor-box occlusion causes no fusion failure on X-Sub in Milestone 3, so "occlusion-robust"
currently rests on skeleton part dropout alone. Two options:

- Keep "occlusion" and add the evidence it needs: the Kinect inferred-joint natural-occlusion split
  and a coupled occlusion (the same box removing the RGB pixels and the covered joints).
- Or retitle to "corruption-robust" and keep occlusion as one corruption family.

## References

Babey, N., Gu, T., Li, Y., Meo, C., & Zhu, K. (2025). *Grounding foundational vision models with 3D human poses for robust action recognition* [Preprint]. arXiv. https://arxiv.org/abs/2511.05622

Bao, W., Yu, Q., & Kong, Y. (2021). Evidential deep learning for open set action recognition. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 13329-13338). https://doi.org/10.1109/ICCV48922.2021.01310

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1-3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

Cao, B., Xia, Y., Ding, Y., Zhang, C., & Hu, Q. (2024). Predictive dynamic fusion. In *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 5608-5628). PMLR. https://proceedings.mlr.press/v235/cao24c.html

Chen, Y., Peng, K., Roitberg, A., Schneider, D., Zhang, J., Zheng, J., Chen, Y., Liu, R., & Yang, K. (2025). Exploring self-supervised skeleton-based action recognition in occluded environments. In *2025 International Joint Conference on Neural Networks* (pp. 1-9). https://doi.org/10.1109/IJCNN64981.2025.11228855

Chen, Y., Zhang, Z., Yuan, C., Li, B., Deng, Y., & Hu, W. (2021). Channel-wise topology refinement graph convolution for skeleton-based action recognition. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 13339-13348). https://doi.org/10.1109/ICCV48922.2021.01311

Corbière, C., Thome, N., Bar-Hen, A., Cord, M., & Pérez, P. (2019). Addressing failure prediction by learning model confidence. In *Advances in Neural Information Processing Systems* (Vol. 32, pp. 2902-2913). https://papers.nips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html

Das, S., Sharma, S., Dai, R., Brémond, F., & Thonnat, M. (2020). VPN: Learning video-pose embedding for activities of daily living. In *Computer Vision - ECCV 2020* (pp. 72-90). Springer. https://doi.org/10.1007/978-3-030-58545-7_5

Dong, H., Zhang, F., Liu, C., & Mu, L. (2026). MAGNet: Enhancing action recognition with multimodal fusion and adaptive graph convolution. *Scientific Reports, 16*, 15100. https://doi.org/10.1038/s41598-026-45601-2

Du, C., Teng, J., Li, T., Liu, Y., Yuan, T., Wang, Y., Yuan, Y., & Zhao, H. (2023). On uni-modal feature learning in supervised multi-modal learning. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 8632-8656). PMLR. https://proceedings.mlr.press/v202/du23e.html

Duan, H., Zhao, Y., Chen, K., Lin, D., & Dai, B. (2022). Revisiting skeleton-based action recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2959-2968). https://doi.org/10.1109/CVPR52688.2022.00298

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In *Proceedings of the 33rd International Conference on Machine Learning* (Vol. 48, pp. 1050-1059). PMLR. https://proceedings.mlr.press/v48/gal16.html

Gao, Z., Jiang, X., Xu, X., Shen, F., Li, Y., & Shen, H. T. (2024). Embracing unimodal aleatoric uncertainty for robust multimodal fusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 26866-26875). https://doi.org/10.1109/CVPR52733.2024.02538

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4878-4887). https://papers.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf

Geifman, Y., Uziel, G., & El-Yaniv, R. (2019). Bias-reduced uncertainty estimation for deep neural classifiers. In *International Conference on Learning Representations*. https://openreview.net/forum?id=SJfb5jCqKm

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321-1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

Guo, H., Ren, Z., Wu, Y., Hua, G., & Ji, Q. (2022). Uncertainty-based spatial-temporal attention for online action detection. In *Computer Vision - ECCV 2022* (pp. 69-86). Springer. https://doi.org/10.1007/978-3-031-19772-7_5

Guo, M. (2026, August 11). Lightweight probability-level fusion for occlusion-robust human activity recognition from RGB and skeleton data. *National High School Journal of Science*. https://nhsjs.com/2026/lightweight-probability-level-fusion-for-occlusion-robust-human-activity-recognition-from-rgb-and-skeleton-data/

Han, Z., Yang, F., Huang, J., Zhang, C., & Yao, J. (2022). Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 20675-20685). https://doi.org/10.1109/CVPR52688.2022.02005

Han, Z., Zhang, C., Fu, H., & Zhou, J. T. (2021). Trusted multi-view classification. In *International Conference on Learning Representations*. https://openreview.net/forum?id=OOsR8BzCnl5

Han, Z., Zhang, C., Fu, H., & Zhou, J. T. (2023). Trusted multi-view classification with dynamic evidential fusion. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*(2), 2551-2566. https://doi.org/10.1109/TPAMI.2022.3171983

Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. In *International Conference on Learning Representations*. https://arxiv.org/abs/1903.12261

Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Hkg4TI9xl

Jaeger, P. F., Lüth, C. T., Klein, L., & Bungert, T. J. (2023). A call to reflect on evaluation practices for failure detection in image classification. In *International Conference on Learning Representations*. https://openreview.net/forum?id=YnkGMIh0gvX

Khanal, A., & Zhou, J. (2026). *Severe domain shift in skeleton-based action recognition: A study of uncertainty failure in real-world gym environments* [Preprint]. arXiv. https://arxiv.org/abs/2603.15574

Korbar, B., Tran, D., & Torresani, L. (2019). SCSampler: Sampling salient clips from video for efficient action recognition. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 6231-6241). https://doi.org/10.1109/ICCV.2019.00633

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 6402-6413). https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html

Liu, C., Jiang, Y., Du, C., & Li, Z. (2024). Enhancing action recognition from low-quality skeleton data via part-level knowledge distillation. *Signal Processing, 221*, 109486. https://doi.org/10.1016/j.sigpro.2024.109486

Liu, C., Liu, X., Yu, Z., Hou, Y., Yue, H., & Yang, J. (2024). Adversarial robustness in RGB-skeleton action recognition: Leveraging attention modality reweighter. In *2024 IEEE International Joint Conference on Biometrics* (pp. 1-11). https://doi.org/10.1109/IJCB62174.2024.10744468

Liu, J., Shahroudy, A., Perez, M., Wang, G., Duan, L.-Y., & Kot, A. C. (2020). NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 42*(10), 2684-2701. https://doi.org/10.1109/TPAMI.2019.2916873

Liu, J., Shahroudy, A., Xu, D., Kot, A. C., & Wang, G. (2018). Skeleton-based action recognition using spatio-temporal LSTM network with trust gates. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 40*(12), 3007-3021. https://doi.org/10.1109/TPAMI.2017.2771306

Ma, H., Zhang, Q., Zhang, C., Wu, B., Fu, H., Zhou, J. T., & Hu, Q. (2023). Calibrating multimodal learning. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 23429-23450). PMLR. https://proceedings.mlr.press/v202/ma23i.html

Ma, M., Ren, J., Zhao, L., Testuggine, D., & Peng, X. (2022). Are multimodal transformers robust to missing modality? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 18177-18186). https://doi.org/10.1109/CVPR52688.2022.01764

Malzard, S., Mital, N., Walters, R., Nockles, V., Rao, R., & De Melo, C. M. (2025). *SHARDeg: A benchmark for skeletal human action recognition in degraded scenarios* [Preprint]. arXiv. https://arxiv.org/abs/2505.18048

Microsoft. (2014). *TrackingState enumeration* [Kinect for Windows SDK 2.0 API reference]. https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn758896(v=ieb.10)

Minderer, M., Djolonga, J., Romijnders, R., Hubis, F., Zhai, X., Houlsby, N., Tran, D., & Lucic, M. (2021). Revisiting the calibration of modern neural networks. In *Advances in Neural Information Processing Systems* (Vol. 34). https://papers.neurips.cc/paper_files/paper/2021/file/8420d359404024567b5aefda1231af24-Paper.pdf

Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence, 29*(1), 2901-2907. https://doi.org/10.1609/aaai.v29i1.9602

Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J., Lakshminarayanan, B., & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In *Advances in Neural Information Processing Systems* (Vol. 32, pp. 13991-14002). https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

Peng, K., Roitberg, A., Yang, K., Zhang, J., & Stiefelhagen, R. (2023). Delving deep into one-shot skeleton-based action recognition with diverse occlusions. *IEEE Transactions on Multimedia, 25*, 1489-1504. https://doi.org/10.1109/TMM.2023.3235300

Peng, X., Wei, Y., Deng, A., Wang, D., & Hu, D. (2022). Balanced multimodal learning via on-the-fly gradient modulation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 8228-8237). https://doi.org/10.1109/CVPR52688.2022.00806

Reilly, D., & Das, S. (2024). Just add π! Pose induced video transformers for understanding activities of daily living. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 18340-18350). https://doi.org/10.1109/CVPR52733.2024.01736

Roitberg, A., Haurilet, M., Martinez, M., & Stiefelhagen, R. (2021). Uncertainty-sensitive activity recognition: A reliability benchmark and the CARING models. In *2020 25th International Conference on Pattern Recognition* (pp. 3814-3821). https://doi.org/10.1109/ICPR48806.2021.9412540

Schiappa, M. C., Biyani, N., Kamtam, P., Vyas, S., Palangi, H., Vineet, V., & Rawat, Y. S. (2023). A large-scale robustness analysis of video action recognition models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 14698-14708). https://doi.org/10.1109/CVPR52729.2023.01412

Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. In *Advances in Neural Information Processing Systems* (Vol. 31, pp. 3179-3189). https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html

Shahroudy, A., Liu, J., Ng, T.-T., & Wang, G. (2016). NTU RGB+D: A large scale dataset for 3D human activity analysis. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1010-1019). https://doi.org/10.1109/CVPR.2016.115

Shao, D., Shi, M., & Liu, L. (2026). FineTec: Fine-grained action recognition under temporal corruption via skeleton decomposition and sequence completion. *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(11), 8842-8850. https://doi.org/10.1609/aaai.v40i11.37838

Shen, T., Xu, H., Ahuja, N., Tickoo, O., Shin, P., & Narayanan, V. (2026). Parts-Mamba: Augmenting joint context with part-level scanning for occluded human skeleton. In *2026 IEEE International Conference on Image Processing* (pp. 1-6). https://doi.org/10.1109/ICIP61757.2026.11630168

Song, Y.-F., Zhang, Z., & Wang, L. (2019). Richly activated graph convolutional network for action recognition with incomplete skeletons. In *2019 IEEE International Conference on Image Processing* (pp. 1-5). https://doi.org/10.1109/ICIP.2019.8802917

Song, Y.-F., Zhang, Z., Shan, C., & Wang, L. (2021). Richly activated graph convolutional network for robust skeleton-based action recognition. *IEEE Transactions on Circuits and Systems for Video Technology, 31*(5), 1915-1925. https://doi.org/10.1109/TCSVT.2020.3015051

Subedar, M., Krishnan, R., Lopez Meyer, P., Tickoo, O., & Huang, J. (2019). Uncertainty-aware audiovisual activity recognition using deep Bayesian variational inference. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 6301-6310). https://doi.org/10.1109/ICCV.2019.00640

Takami, M., & Fukuda, T. (2026). Confidence-aware gated multimodal fusion for robust temporal action localization in occluded environments. *Sensors, 26*, 2454. https://doi.org/10.3390/s26082454

Tanaka, N., Kera, H., & Kawamoto, K. (2024). Fourier analysis on robustness of graph convolutional neural networks for skeleton-based action recognition. *Computer Vision and Image Understanding, 240*, 103936. https://doi.org/10.1016/j.cviu.2024.103936

Tomani, C., Cremers, D., & Buettner, F. (2022). Parameterized temperature scaling for boosting the expressive power in post-hoc uncertainty calibration. In *Computer Vision - ECCV 2022* (pp. 555-569). Springer. https://doi.org/10.1007/978-3-031-19778-9_32

Tomani, C., Gruber, S., Erdem, M. E., Cremers, D., & Buettner, F. (2021). Post-hoc uncertainty calibration for domain drift scenarios. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 10119-10127). https://doi.org/10.1109/CVPR46437.2021.00999

Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). A closer look at spatiotemporal convolutions for action recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 6450-6459). https://doi.org/10.1109/CVPR.2018.00675

Wang, H., He, F., Peng, Z., Shao, T., Yang, Y.-L., Zhou, K., & Hogg, D. (2021). Understanding the robustness of skeleton-based action recognition under adversarial attack. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 14651-14660). https://doi.org/10.1109/CVPR46437.2021.01442

Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X., & Van Gool, L. (2016). Temporal segment networks: Towards good practices for deep action recognition. In *Computer Vision - ECCV 2016* (pp. 20-36). Springer. https://doi.org/10.1007/978-3-319-46484-8_2

Wang, W., Tran, D., & Feiszli, M. (2020). What makes training multi-modal classification networks hard? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 12692-12702). https://doi.org/10.1109/CVPR42600.2020.01271

Wei, S., Zhang, K., Chen, L., He, T., & Duan, G. (2026). Unbiased dynamic multimodal fusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 6242-6252). https://doi.org/10.1109/CVPRF71909.2026.00599

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1

Woo, S., Lee, S., Park, Y., Nugroho, M. A., & Kim, C. (2023). Towards good practices for missing modality robust action recognition. *Proceedings of the AAAI Conference on Artificial Intelligence, 37*(3), 2776-2784. https://doi.org/10.1609/aaai.v37i3.25378

Wu, N., Jastrzębski, S., Cho, K., & Geras, K. J. (2022). Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 24043-24055). PMLR. https://proceedings.mlr.press/v162/wu22d.html

Xu, Z., Tang, C., Hu, H., Wang, W., Qiao, S., & Tong, A. (2025). Attention mechanism based multimodal feature fusion network for human action recognition. *Journal of Visual Communication and Image Representation, 110*, 104459. https://doi.org/10.1016/j.jvcir.2025.104459

Yang, M., Li, Y., Zhang, C., Hu, P., & Peng, X. (2024). Test-time adaptation against multi-modal reliability bias. In *International Conference on Learning Representations*. https://iclr.cc/virtual/2024/poster/18574

Yi, C., Yang, S., Li, H., Tan, Y.-P., & Kot, A. C. (2021). Benchmarking the robustness of spatial-temporal models against corruptions. In *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2a38a4a9316c49e5a833517c45d31070-Abstract-round2.html

Yoon, Y., Yu, J., & Jeon, M. (2022). Predictively encoded graph convolutional network for noise-robust skeleton-based action recognition. *Applied Intelligence, 52*(3), 2317-2331. https://doi.org/10.1007/s10489-021-02487-z

Yu, B. X. B., Liu, Y., Zhang, X., Zhong, S.-h., & Chan, K. C. C. (2023). MMNet: A model-based multimodal network for human action recognition in RGB-D videos. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*(3), 3522-3538. https://doi.org/10.1109/TPAMI.2022.3177813

Zhang, Q., Wu, H., Zhang, C., Hu, Q., Fu, H., Zhou, J. T., & Peng, X. (2023). Provable dynamic fusion for low-quality multimodal data. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 41753-41769). PMLR. https://proceedings.mlr.press/v202/zhang23ar.html

Zhang, Q., Wei, Y., Han, Z., Fu, H., Peng, X., Deng, C., Hu, Q., Xu, C., Wen, J., Hu, D., & Zhang, C. (2026). Multimodal fusion on low-quality data: A comprehensive survey. *Information Fusion, 135*, 104437. https://doi.org/10.1016/j.inffus.2026.104437

Zhu, F., Cheng, Z., Zhang, X.-Y., & Liu, C.-L. (2022). Rethinking confidence calibration for failure prediction. In *Computer Vision - ECCV 2022* (pp. 518-536). Springer. https://doi.org/10.1007/978-3-031-19806-9_30
