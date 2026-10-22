# Experiment: frozen ECAPA speaker encoder (R=6)

**Question:** do wrong-speaker errors on similar voices drop if the jointly trained ResNet
speaker encoder is replaced by a pretrained, frozen ECAPA-TDNN?

**Why we expect it to help** (measured 2026-10-22 on the 40 unseen Libri2Mix test speakers,
15 utterances each, 6 s center crops):

| speaker encoder | verification EER |
|---|---|
| ResNet inside `saved_models/multispk_r6_ep79` (trained on 251 speakers) | 6.1% |
| ECAPA-TDNN (SpeechBrain `spkrec-ecapa-voxceleb`), on 8 kHz audio upsampled to 16 kHz | 2.0% (3 s crops: 2.9%) |

The R=6 model's wrong-speaker rate roughly doubles for the most similar 20% of
target/interferer pairs: on 2 speakers, 5.6% for the least similar pairs vs 11.1% for the
most similar.

**Setup:** `config.yaml` is `configs/multispk_deep.yaml` with only `model.spk_encoder:
ecapa_frozen`:

* the same R=6 separator, loss, 2–4 speaker dynamic mixing and validation sets;
* the same LR schedule, 80 epochs, trained from scratch.

The ECAPA embedding (192-d) is L2-normalised and passed through a trainable Linear to 256-d.
The speaker-classification loss (weight 0.5) still trains that projection.

Parameters:

* 30.9M in total, of which 20.8M are the frozen ECAPA weights.
* 10.1M are trainable.
* For comparison, the ResNet model has 13.6M, all trainable.

**First attempt, stopped after 3 epochs:** `r6_ecapa_unscaled_stopped/` (logs only). In that
run the L2-normalised ECAPA embedding entered the projection with norm 1. That made the speaker
embedding about 3.6x smaller at init than the ResNet's. FiLM weights are zero-initialised and
their gradient is proportional to the embedding, so the speaker cue was learned very slowly.

| | val SI-SDRi (mean 2/3/4 spk) | wrong-speaker rate |
|---|---|---|
| ECAPA, epochs 1–3 | 1.70 / 1.76 / 1.63 dB | 18% → 31% |
| ResNet run, epochs 1–3 | 2.67 / 3.35 / 3.76 dB | about 10–12% |

The fix scales the normalised embedding by sqrt(192), giving unit RMS per dimension. Training
was restarted from scratch at 10:46 on 2026-10-22.

**Run:** `powershell -ExecutionPolicy Bypass -File experiments\ecapa_frozen\run.ps1`. It is
resumable, and outputs go to `experiments/ecapa_frozen/r6_ecapa/`. Compare against the
ResNet model with:

    python scripts/compare_results.py saved_models/multispk_r6_ep79 experiments/ecapa_frozen/r6_ecapa

The code is shared with the main project (`src/tse/models/speaker_encoder.py`). The
pretrained ECAPA is downloaded once into `pretrained_models/` and needs `pip install
speechbrain`, tested with 1.1.2.
