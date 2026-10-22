# U-eConv-TSE: Target Speaker Extraction on Libri2Mix

Training and evaluation pipeline for **target speaker extraction** (TSE): the model takes a
2-speaker mixture plus an enrollment utterance of the target speaker and outputs only that
speaker's speech. There is no PIT anywhere.

* **Proposed:** `UEConvTSE`. U-ConvBlock stages with FiLM speaker conditioning at the
  coarsest scale, a Gated Embedding Split (GES) into target and interference streams, and
  Multi-stage Gated Feature Aggregation (MGFA).
* **Baseline:** `TDSpeakerBeam`. A Conv-TasNet separator with multiplicative adaptation.
  It uses the same encoder, decoder, speaker encoder and training recipe as the proposed model.

## Setup

```bash
cd tse_project
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs cu128+
pip install -r requirements.txt
```

Tested with Python 3.13.7, torch 2.11.0+cu128 and an RTX 5090 (32 GB) on Windows 11.
`pesq` has no wheel for Python 3.13 on Windows, so PESQ is skipped automatically. `fast_bss_eval`
(SDR) and `pystoi` (STOI) are used when installed.

**Dataset.** Point `data.root` (in `configs/proposed.yaml`, relative to `tse_project/`) or
`--data-root` at the Libri2Mix folder that contains `wav8k/`. The default is `../data/Libri2Mix`.
The dataset is only read, never modified. Speaker gender comes from LibriSpeech
`SPEAKERS.TXT`, which is looked up in `<root>/` and `<root>/../LibriSpeech/`, or set with
`data.gender_file`.

## Layout

```
tse_project/
  configs/            proposed.yaml, baseline_tdspeakerbeam.yaml (base: proposed), debug.yaml
  src/tse/
    config.py         defaults for every key, YAML `base:` inheritance, --set overrides
    data/index.py     metadata -> per-split index (paths rebuilt from root + mixture ID), cache, custom split
    data/enrollment.py speaker -> utterance pool, 2 samples per mixture, fixed eval enrollments
    data/dataset.py   TSEDataset, collate, prepare_splits()
    models/layers.py  gLN, DS-conv, FiLM / mult / concat fusion, SE attention, encoder/decoder
    models/ublock.py  UConvBlock (FiLM at bottleneck), TCN block / TCN stage
    models/ges_mgfa.py GES and MGFA
    models/speaker_encoder.py ResNet speaker encoder (+ optional frozen ECAPA)
    models/ueconv_tse.py, models/tdspeakerbeam.py
    losses.py metrics.py engine.py utils.py
  scripts/            prepare_data, train, evaluate, infer, profile_model, overfit_check, run_ablations.sh
  tests/              pytest suite (synthetic Libri2Mix tree; no real data needed)
  cache/              cached indexes, enrollment_{split}_*.csv, train_speakers.json  (generated)
  runs/<name>/        config.yaml, train_log.csv, train.log, tb/, best.pt, last.pt, test_metrics.json
```

## Commands

Run these from `tse_project/` with the venv's python. Relative config paths resolve against
`tse_project/`, so they also work from any other directory.

```bash
python -m pytest -q                                              # unit tests
python scripts/prepare_data.py --verify                          # build caches + print stats
python scripts/overfit_check.py --config configs/proposed.yaml   # sanity: must gain > 5 dB
python scripts/overfit_check.py --config configs/baseline_tdspeakerbeam.yaml

# smoke test (tiny model, 64 samples, 2 epochs)
python scripts/train.py --config configs/debug.yaml
python scripts/evaluate.py --ckpt runs/debug/best.pt --max-samples 20 --out-dir runs/debug/eval_tiny

# full training
python scripts/train.py --config configs/proposed.yaml
python scripts/train.py --config configs/baseline_tdspeakerbeam.yaml
python scripts/train.py --config configs/proposed.yaml --resume            # continue from last.pt

# evaluation on the full test set (6000 samples, fixed enrollments)
python scripts/evaluate.py --ckpt runs/proposed/best.pt
python scripts/evaluate.py --ckpt runs/baseline_tdspeakerbeam/best.pt

# ablations (train + evaluate each; finished runs are skipped, interrupted runs resume)
bash scripts/run_ablations.sh
MAX_EPOCHS=30 ONLY="abl_no_mgfa abl_no_ges_int" bash scripts/run_ablations.sh

# inference on any wav (resampled to 8 kHz, mono, any length)
python scripts/infer.py --ckpt runs/proposed/best.pt --mix mix.wav --enroll enroll.wav --out out.wav

# profiling (params, GFLOPs via FlopCounterMode, peak GPU memory, latency; 4 s input)
python scripts/profile_model.py
```

Override any config key from the command line, for example:
`--set model.R=4 train.lr=5e-4 train.amp=bf16`. Unknown keys raise an error.

## Data pipeline

* **Splits:** `train-100` is used for training, `dev` for validation and checkpoint
  selection, and `test` for the final report. They are speaker-disjoint (251 / 40 / 40
  speakers), and `prepare_data.py` checks this. If the dev/test metadata are missing,
  `data/index.py` creates a speaker-disjoint 80/10/10 split by speaker with a fixed seed
  and writes it to `splits/*.csv`.
* **Samples:** each mixture gives two samples, with target = s1 and target = s2.
  `interference = mixture - target`.
* **Enrollment pool:** each split gets its own `speaker -> {utterance_id -> clean s1/s2 file}`
  map, deduplicated by LibriSpeech utterance ID. The enrollment is always the same speaker,
  a different utterance ID, and the same split. In training it is re-sampled on every access
  (torch RNG, seeded per worker). For dev/test it is drawn once with `data.enroll_seed` and
  saved to `cache/enrollment_{split}_8000_min_clean.csv`, so every model sees identical
  pairs. Samples whose speaker has no other utterance are dropped and counted; on Libri2Mix
  none are dropped.
* **Audio:** soundfile, float32, 8 kHz. Training uses a random 3 s crop (the same crop for
  mixture and target, zero-padded if shorter) and a random 4 s enrollment crop (tiled if
  shorter). Evaluation uses the full-length mixture with batch size 1 and a 6 s center crop
  of the enrollment (tiled if shorter).

Dataset statistics from `prepare_data.py` on this machine:

| split | mixtures | samples | dropped | speakers | unique utts | length min / median / max (s) | hours |
|---|---|---|---|---|---|---|---|
| train-100 | 13900 | 27800 | 0 | 251 | 27800 | 3.00 / 12.48 / 16.60 | 43.3 |
| dev | 3000 | 6000 | 0 | 40 | 2250 | 3.00 / 4.80 / 17.46 | 4.5 |
| test | 3000 | 6000 | 0 | 40 | 2073 | 3.00 / 4.59 / 13.99 | 4.2 |

## Models

The default hyper-parameters are `N=512, L=16, stride=8, B=128, H=512, Q=4, K=5, R=3,
spk_dim=256`.

**UEConvTSE** runs `encoder -> gLN -> 1x1 (N->B)`, then R stages. Each stage is
`UConvBlock(x, e)`: 1x1 B->H, PReLU, gLN, then Q stride-2 depthwise-separable conv blocks
(PReLU, gLN). At T/16 it applies FiLM `gamma(e)*x + beta(e)` (identity at init) followed by
SE channel attention. The expanding path uses nearest upsampling plus skip sums, then
1x1 H->B and a residual add. Next comes `GES(x_r, e)`: `glu = a*sigmoid(g + W e)` ->
1x1 -> GroupNorm(2, 2B) -> `F_t, F_i`. MGFA then computes `E_r = beta*E_{r-1} +
(1-beta)*F_r`. The output is `mask = ReLU(1x1(PReLU(E_t_R)))` followed by the decoder.
The interference mask and decoder run only when `compute_interference=True`, which is
training only. At inference GES computes only the target half (exactly equal) and the
interference MGFA and mask are skipped.

**Speaker encoder (`resnet`)** runs the shared encoder, gLN, a 1x1 conv to 256 channels,
then three residual conv blocks (BatchNorm, PReLU, max-pool 3) with channels
256->256->512->512, mean pooling over time and a Linear to `spk_dim`. A training-only
classifier `Linear(spk_dim, 251)` is used for the CE loss.

**TDSpeakerBeam** has a bottleneck followed by 3 repeats of 8 dilated TCN blocks (P=3,
non-causal, gLN, residual). The output of TCN block 7 (0-based, the end of the first
repeat) is multiplied by `Linear(e)`.

## Loss and training

`L = -SI-SDR(est, target) + lambda_ce * CE(logits, label) + lambda_int * -SI-SDR(interf_est, mix - target)`,
with `lambda_ce = 0.5` and `lambda_int = 0.1` (0 for the baseline, which has no interference
branch). SI-SDR is computed in float32 outside autocast, zero-mean, with `eps = 1e-8`.
Steps with a non-finite loss or gradient are skipped and logged.

The optimizer is Adam (lr 1e-3, wd 0) with batch size 8 and gradient clipping at 5.
ReduceLROnPlateau (×0.5, patience 3) and early stopping (patience 12) both track validation
SI-SDRi; training runs for at most 100 epochs. bf16 autocast is optional. One epoch is one
random 3 s crop of each of the 27,800 samples, which is 3,475 steps. Validation runs every
epoch on a fixed, seeded 500-sample subset of dev at full length and reports SI-SDR,
SI-SDRi and NSR.

## Config keys

| key | default | meaning |
|---|---|---|
| `seed`, `deterministic` | 42, false | seeding; deterministic cuDNN/algorithms |
| `data.root` | `../data/Libri2Mix` | Libri2Mix root (also `--data-root`) |
| `data.sample_rate`, `data.mode`, `data.mix_type` | 8000, min, clean | `wav8k/min`, `mix_clean` (`both` = `mix_both`) |
| `data.train_split/val_split/test_split` | train-100 / dev / test | split folder names |
| `data.segment_seconds` | 3.0 | training crop |
| `data.enroll_seconds_train/eval` | 4.0 / 6.0 | enrollment length |
| `data.enroll_seed` | 1234 | fixed dev/test enrollment draw |
| `data.val_subset` | 500 | validation samples per epoch |
| `data.max_train_samples` | null | subsample training samples (debug) |
| `data.cache_dir`, `data.splits_dir`, `data.gender_file` | cache, splits, auto | |
| `model.type` | ueconv_tse | `ueconv_tse` or `tdspeakerbeam` |
| `model.N, L, stride, B, H, Q, K, R, spk_dim` | 512, 16, 8, 128, 512, 4, 5, 3, 256 | sizes |
| `model.separator` | ublock | `ublock` or `tcn` (FiLM at each stage input + X TCN blocks) |
| `model.fusion` | film | `film`, `mult` or `concat` |
| `model.inject` | every_stage | `every_stage` or `input_only` |
| `model.use_ges_interference` | true | interference branch + auxiliary loss |
| `model.use_mgfa`, `model.mgfa_beta` | true, channel | MGFA on/off; `scalar` or `channel` beta |
| `model.upsample` | nearest | `nearest` or `transposed` |
| `model.mask_act` | relu | `relu` or `sigmoid` |
| `model.se_reduction` | 8 | SE attention reduction at the bottleneck |
| `model.P, X, adapt_layer` | 3, 8, 7 | TCN settings (baseline and `separator=tcn`) |
| `model.spk_encoder`, `share_encoder`, `spk_channels` | resnet, true, [256,512,512] | speaker encoder |
| `loss.lambda_ce`, `loss.lambda_int` | 0.5, 0.1 | loss weights |
| `train.batch_size, lr, weight_decay, grad_clip` | 8, 1e-3, 0, 5.0 | |
| `train.max_epochs, lr_factor, lr_patience, early_stop_patience` | 100, 0.5, 3, 12 | |
| `train.amp` | none | `none` or `bf16` (forward only) |
| `train.num_workers, log_every, max_steps_per_epoch, device` | 4, 100, null, auto | |

## Evaluation outputs

`evaluate.py` writes `test_metrics.json` and `test_per_utt.csv` next to the checkpoint.
`test_metrics.json` contains, overall and by same versus different gender: SI-SDR out and
in, SI-SDRi (mean, median, std), NSR (fraction with SI-SDRi < 0), the fraction with
SI-SDRi < 5 dB, SDRi (`fast_bss_eval`), STOI, and PESQ (narrowband) if `pesq` is installed.
The per-utterance CSV has the columns `mixture_id, target_role, speaker_id, enrollment_path,
length_s, si_sdr_in, si_sdr_out, si_sdri, same_gender`, followed by the extra metrics.
On the RTX 5090, the full 6,000-sample test takes about 22 minutes with SDR and STOI, and
about 3 minutes with `--no-extra`.

## Multi-speaker (2–4 speakers) extension

The original model was trained only on 2-speaker mixtures. A quick test (`scripts/demo_3spk.py`)
showed that it still runs on 3-speaker mixtures, but with much more leakage from the other
voices. This extension adds 3- and 4-speaker evaluation sets and fine-tuning on mixtures of
2–4 speakers.

* **Saved 2-speaker model:** `saved_models/proposed_2spk_ep74/`. The `.pt` files are
  read-only.
  * `best.pt` is from epoch 74, with 10.24 dB val SI-SDRi.
  * `last.pt` is from epoch 80.
  * The folder also holds the config, logs, TensorBoard logs, the fixed enrollment tables,
    `snapshot_info.json` (SHA-256 checksums) and its test metrics.
* **Evaluation sets:** `scripts/make_multispk_sets.py` writes them to `generated/`, about
  3.6 GB in total.
  * `Libri3Mix` dev/test is the official benchmark, rebuilt from the LibriMix metadata exactly
    as the official generator does it (gain at 16 kHz, `resample_poly` to 8 kHz, truncation
    to the shortest source). A check against the official Libri2Mix files sharing the same
    utterances matched to within one 16-bit step.
  * `Libri4Mix` dev/test is our own set, not an official benchmark. It takes each Libri3Mix
    mixture and adds a fourth speaker from the same LibriSpeech subset, chosen with a fixed
    seed. Its level is uniform between the quietest and loudest of the other three. The
    recipe is saved in `metadata/libri4mix_*_recipe.csv`.
  * Each set gets its own fixed enrollment table, the same way as Libri2Mix.
* **The index is N-speaker:** it reads the number of sources from the mixture ID. Each
  mixture gives one sample per source. For 3 or more speakers, `same_gender` means at least
  one interferer has the target's gender.
* **Dynamic mixing for training** (`data.dynamic_mix`, `tse/data/dynamic_mix.py`). Each
  training item is a fresh mixture:
  * the target utterance plus 1–3 interferers from other training speakers;
  * 2, 3 or 4 speakers with probabilities 0.40 / 0.35 / 0.25;
  * an independent level per source, U(−5, 5) dB;
  * with probability 0.25 an interferer covers only 50–100% of the 3 s segment.

  The sources are the clean train-100 utterances, so the train split stays
  speaker-disjoint from dev/test.
* **Fine-tuning** (`configs/finetune_multispk.yaml`):
  * starts from `saved_models/proposed_2spk_ep74/best.pt` via `train.init_from` (weights
    only, fresh optimizer);
  * lr 2.5e-4 for a fixed budget of 25 epochs;
  * every epoch validates on 400 samples each of 2-, 3- and 4-speaker dev, and
    `best.pt` / the LR schedule use the mean SI-SDRi over the three sets.

  The architecture and loss are unchanged, so the effect comes only from the training data.

```bash
python scripts/make_multispk_sets.py --sets 3 4 --splits dev test
python scripts/train.py --config configs/finetune_multispk.yaml       # or scripts/run_finetune.ps1
python scripts/evaluate.py --ckpt runs/finetune_multispk/best.pt                                   # 2 speakers
python scripts/evaluate.py --ckpt runs/finetune_multispk/best.pt --eval-root generated/Libri3Mix   # 3 speakers
python scripts/evaluate.py --ckpt runs/finetune_multispk/best.pt --eval-root generated/Libri4Mix   # 4 speakers
python scripts/demo_3spk.py --ckpt runs/finetune_multispk/best.pt --out-dir demo_3spk_finetuned
```

New config keys:

| key | default | meaning |
|---|---|---|
| `data.dynamic_mix.enabled` | false | on-the-fly 2–4 speaker mixing for training |
| `data.dynamic_mix.n_spk_probs` | [0.4, 0.35, 0.25] | P(2, 3, 4 speakers), target included |
| `data.dynamic_mix.level_db_range` | [-5, 5] | per-source level range (dB) |
| `data.dynamic_mix.partial_overlap_prob`, `min_overlap` | 0.25, 0.5 | partial-overlap interferers |
| `data.extra_val_sets` | [] | extra validation sets `{name, root, split, subset}` |
| `data.val_name` | 2spk | log name of the main validation set |
| `train.init_from` | null | initialise weights from a checkpoint |

### Multi-speaker results (test sets, measured)

All numbers come from `scripts/evaluate.py` on the full test sets, each with its fixed enrollment table:

* Libri2Mix: 6,000 samples
* Libri3Mix: 9,000 samples
* Libri4Mix: 12,000 samples

Columns:

* **NSR:** the share of outputs with SI-SDRi below 0 dB (wrong speaker or a failed extraction).
* **<5dB:** the share of outputs that improve by less than 5 dB.

To regenerate the table: `python scripts/compare_results.py saved_models/proposed_2spk_ep74 runs/finetune_multispk runs/multispk_deep_r6`.

The models compared:

* **2-spk model:** R=3, 8.7M params, 20.7 GFLOPs per 4 s.
* **Fine-tuned:** the 2-spk model, fine-tuned for 25 epochs on 2–4 speaker dynamic mixing (`configs/finetune_multispk.yaml`).
* **R=6 from scratch:** the same recipe trained from scratch with 6 stages, 13.6M params, 31.0 GFLOPs, for 80 epochs (`configs/multispk_deep.yaml`).

Measured on the RTX 5090 with fp32 at batch 1, the latency for 4 s of audio is 6.3 ms for R=3 and 9.7 ms for R=6.

| test set | model | SI-SDRi | median | SDRi | STOI | NSR % | <5dB % |
|---|---|---|---|---|---|---|---|
| 2 spk (Libri2Mix) | 2-spk model (epoch 74) | **9.61** | 10.98 | 10.12 | 0.858 | **5.6** | **12.0** |
| 2 spk (Libri2Mix) | fine-tuned on 2–4 spk | 8.73 | 10.48 | 9.41 | 0.841 | 8.8 | 17.0 |
| 2 spk (Libri2Mix) | R=6 from scratch, 2–4 spk | 9.49 | 11.11 | 10.15 | 0.854 | 7.5 | 13.0 |
| 3 spk (Libri3Mix) | 2-spk model (epoch 74) | 4.84 | 5.36 | 4.97 | 0.703 | **7.1** | 46.1 |
| 3 spk (Libri3Mix) | fine-tuned on 2–4 spk | 7.19 | 8.49 | 7.78 | 0.734 | 8.3 | 22.4 |
| 3 spk (Libri3Mix) | R=6 from scratch, 2–4 spk | **8.13** | 9.60 | 8.76 | 0.756 | 7.8 | **17.6** |
| 4 spk (Libri4Mix*) | 2-spk model (epoch 74) | 3.13 | 3.40 | 3.20 | 0.602 | 8.9 | 76.9 |
| 4 spk (Libri4Mix*) | fine-tuned on 2–4 spk | 6.06 | 7.05 | 6.62 | 0.653 | **8.5** | 30.3 |
| 4 spk (Libri4Mix*) | R=6 from scratch, 2–4 spk | **7.00** | 8.33 | 7.61 | 0.679 | **8.5** | **23.9** |

\* Our own 4-speaker extension, not an official benchmark.

SI-SDRi split by gender (same gender means at least one interferer has the target's gender):

| test set | 2-spk model: same / different | fine-tuned: same / different | R=6: same / different |
|---|---|---|---|
| 2 spk | 8.59 / 10.62 | 7.87 / 9.59 | 8.55 / 10.42 |
| 3 spk | 4.35 / 6.29 | 6.72 / 8.57 | 7.65 / 9.52 |
| 4 spk | 2.96 / 4.27 | 5.83 / 7.66 | 6.75 / 8.73 |

What the results show:

* **Fine-tuning helps 3–4 speakers a lot.** On 3 speakers it adds +2.35 dB, and outputs below 5 dB drop from 46% to 22%. On 4 speakers it adds +2.93 dB, with 77% → 30%.
* **The cost is on 2 speakers:** −0.88 dB, and NSR rises from 5.6% to 8.8%.
* **Target confusion did not improve.** NSR is roughly flat or worse, while leakage dropped sharply. The remaining errors are mostly same-gender confusions, so a stronger speaker cue is the next lever, not more separation capacity alone.
* **Training stopped early.** The fine-tune was still improving slowly at epoch 25 (val mean 7.41 dB, its best), so the fixed budget limited it.
* **R=6 from scratch is the best multi-speaker model.**
  * Versus the fine-tune: +0.94 dB on 3 speakers and +0.94 dB on 4 speakers. Outputs below 5 dB fall from 22% to 18% on 3 speakers and from 30% to 24% on 4 speakers.
  * It almost recovers 2-speaker quality: 9.49 vs 9.61 dB for the 2-spk-only model, and a higher median (11.11 vs 10.98).
  * Its best validation score came at epoch 79 of 80, with the LR halved at epochs 55 and 63. It had not converged, so a longer schedule would likely add more.
* **Caveat on the R=6 comparison:** it changes two things at once versus the fine-tune — depth (3 → 6 stages) and training regime (80 epochs from scratch vs 25 fine-tuning epochs on a 2-spk model). We have no R=3 run trained from scratch for 80 epochs on 2–4 speakers, so the gain cannot be attributed to depth alone.
* **Wrong-target errors are unchanged across all three models:** NSR is about 8–9% on 3–4 speakers, and about 8–10% for same-gender mixtures. More separation capacity reduced leakage but not target confusion, so the next step should strengthen the speaker cue (e.g. a pretrained ECAPA embedding or a speaker-consistency loss on the output).
* **Recommendation:** use `runs/multispk_deep_r6/best.pt` as the general 2–4 speaker model. Keep `saved_models/proposed_2spk_ep74` only if mixtures are known to have exactly 2 speakers and the lowest 2-speaker NSR matters (5.6% vs 7.5%).

The listening demo for the fine-tuned model is in `demo_3spk_finetuned/index.html`. It uses 300 paired samples: the same Libri2Mix test sample, with and without an added third speaker.

* 2 speakers: 8.95 dB SI-SDRi.
* 3 speakers: 7.29 dB.
* The original model scored 9.99 dB and 5.28 dB on the same protocol.
* The R=6 model's demo is in `demo_3spk_deep_r6/index.html`: 9.74 dB on 2 speakers and 8.48 dB on 3 speakers.

## Interactive web demo

```bash
python scripts/web_demo.py          # or scripts/run_web_demo.ps1; opens http://127.0.0.1:7860
```

How the demo works:

* **Inputs:** for each of 2–4 speakers, upload a file (any format the browser can decode) or record from the microphone. The "Random speakers" button instead draws LibriSpeech test speakers, with an optional same-gender or mixed-gender constraint.
* **Target and reference:** pick the target and give a reference clip of that person. The reference must be a different recording; random speakers come with a second utterance.
* **Mixing:** each source is scaled to the same loudness and then shifted by its level slider (±10 dB). The sources are summed, either padded to the longest or cropped to the shortest.
* **Extraction:** the chosen model extracts the target.
* **Output:** the page plays the mixture, the extracted output and the clean target. It shows SI-SDR in, SI-SDR out and SI-SDRi, plus the output's SI-SDR against every speaker. That last score shows whether the model picked the wrong person.

Implementation:

* The server uses only the Python standard library and binds to localhost.
* The browser resamples audio to 8 kHz mono before sending it.
* Recordings are limited to 60 s each.
* The core logic is in `src/tse/demo_app.py` (tested in `tests/test_demo_app.py`), and the page is `webui/index.html`.

## Deviations from the spec

1. **The metadata paths are stale.** The LibriMix CSVs contain absolute paths from another
   machine (`C:\Users\admin\Desktop\TSE\...`). Paths are therefore rebuilt as
   `root/wav8k/min/{split}/{mix_clean|s1|s2}/{mixture_ID}.wav`, and only `mixture_ID` and
   `length` are read from the CSVs.
2. **`mix_both` and `noise` are not present** in this copy of Libri2Mix. `mix_type: both` is
   implemented but raises a clear `FileNotFoundError` here.
3. **PESQ is skipped** because the package cannot be built on Windows/Python 3.13 without a
   compiler. This is noted in `test_metrics.json`.
4. **`ecapa_frozen` is implemented but untested.** speechbrain is not installed. Selecting it
   raises a clear ImportError, and `run_ablations.sh` skips that ablation.
5. **MGFA `scalar` beta** is `sigmoid(logit)` with logit initialised to 0 (beta = 0.5), the
   same parameterisation as `channel`, so beta stays in (0, 1).
6. **`inject: input_only`** removes all speaker conditioning from stages 2..R. This covers
   both FiLM and the speaker term in the GES gate, so the speaker information enters only at
   stage 1.
7. **The baseline TCN** uses residual connections only, without Conv-TasNet's separate skip
   path. `adapt_layer=7` is a 0-based index over the 24 blocks, which places it at the end of
   the first repeat.
8. **Speaker-encoder residual blocks** use kernel-3 convolutions with BatchNorm (SpEx+-style).
   The spec did not fix the normalisation.
9. **cuDNN benchmark mode** is on for training (fixed 3 s crops) and off for
   validation/evaluation. Variable-length utterances would otherwise trigger autotuning for
   every new length; the first validation pass went from 44 s to under 1 s on 16 utterances.
10. **The validation subset** is a seeded random 500 of the 6,000 dev samples, not the first
    500, which would cover only a few mixtures and speakers.

## Known limitations

* **Short eval enrollments.** In `min` mode every source file is truncated to its mixture's
  length. Dev/test mixtures are short (median 4.6 to 4.8 s), so most 6 s eval enrollments are
  tiled repeats of a 3 to 6 s utterance. The original LibriSpeech FLACs (in
  `../data/LibriSpeech`) would give longer, untruncated enrollments; that is a sensible
  follow-up but not what the spec asked for.
* **Reused dev/test utterances.** Dev/test reuse utterances across mixtures (2,250 / 2,073
  unique out of 6,000 slots), so the same enrollment utterance can serve several samples.
* **The proposed separator is shallow.** With R=3 there are only 3 U-blocks; SuDoRM-RF uses
  16. Its separator compute is about 2.3x lower than the 24-block TCN baseline (11.5 vs
  26.6 GFLOPs per 4 s), so R is the first knob to sweep if it trails the baseline.
* **The speaker encoder is costly.** It accounts for about 9.2 GFLOPs per 4 s of enrollment
  (the residual blocks run at the full 1 kHz frame rate). It only runs once per enrollment,
  so the embedding can be cached in deployment.
* **No automatic OOM reduction.** Measured peak training memory at 3 s crops is 3.4 GB for
  the proposed model at bs 8 (13.2 GB at bs 32) and 12.6 GB for the baseline at bs 8
  (25.1 GB at bs 16; bs 32 does not fit in 32 GB). On Windows the driver may spill to system
  RAM instead of raising OOM, which makes training extremely slow rather than crashing.
* **bf16 needs CUDA.** bf16 autocast on CPU is disabled, because oneDNN on this CPU does not
  support bf16 backward.
