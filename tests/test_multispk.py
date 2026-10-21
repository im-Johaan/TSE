"""N-speaker indexing, dynamic mixing and fine-tuning config tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from tse.config import PROJECT_ROOT, load_config, resolve_paths
from tse.data.dataset import TSEDataset, prepare_eval_set, prepare_splits
from tse.data.dynamic_mix import DynamicMixDataset, active_rms


def test_three_speaker_index_and_samples(fake_libri3mix) -> None:
    d = fake_libri3mix["data_cfg"]
    samples = prepare_eval_set(d, d["root"], "test")
    assert len(samples) == 3 * fake_libri3mix["n_mixtures"]
    assert set(samples["role"]) == {"s1", "s2", "s3"}
    assert (samples["n_src"] == 3).all()
    for r in samples.itertuples(index=False):
        others = r.other_speaker_id.split("+")
        assert len(others) == 2 and r.speaker_id not in others
        assert r.enroll_utt.split("-")[0] == r.speaker_id and r.enroll_utt != r.utt_id
        # same_gender: at least one interferer shares the target's gender (odd ids = F)
        g = int(r.speaker_id) % 2
        assert r.same_gender == str(any(int(o) % 2 == g for o in others))


def test_three_speaker_dataset_items(fake_libri3mix) -> None:
    d = fake_libri3mix["data_cfg"]
    samples = prepare_eval_set(d, d["root"], "test")
    ds = TSEDataset(samples, d["root"], None, {}, train=False, enroll_seconds=2.0)
    item = ds[0]
    assert item["meta"]["n_src"] == 3
    # interference = sum of the other two sources (= mixture - target, up to 16-bit rounding)
    assert torch.allclose(item["interference"], item["mixture"] - item["target"])


def test_eval_set_enrollments_deterministic(fake_libri3mix, tmp_path) -> None:
    d = dict(fake_libri3mix["data_cfg"])
    a = prepare_eval_set(d, d["root"], "test")
    b = prepare_eval_set(dict(d, cache_dir=str(tmp_path / "other")), d["root"], "test")
    cols = ["mixture_id", "role", "enroll_utt"]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def _dm_dataset(fake_librimix, **kw) -> tuple[DynamicMixDataset, dict]:
    d = fake_librimix["data_cfg"]
    data = prepare_splits(d)
    ds = DynamicMixDataset(data.samples["train"], d["root"], data.pools["train"], data.spk2label,
                           segment_seconds=0.5, enroll_seconds=1.0, **kw)
    return ds, data


def test_dynamic_mix_items(fake_librimix) -> None:
    ds, data = _dm_dataset(fake_librimix, n_spk_probs=[0.0, 1.0])  # always 3 speakers
    torch.manual_seed(0)
    for i in range(len(ds)):
        item = ds[i]
        meta = item["meta"]
        assert meta["n_src"] == 3
        assert item["mixture"].shape == (4000,) and item["enrollment"].shape == (8000,)
        assert torch.allclose(item["interference"], item["mixture"] - item["target"])
        assert float(item["mixture"].abs().max()) <= 0.95 + 1e-5
        interferer_spk = [u.split("-")[0] for u in meta["mixture_id"][3:].split("_")[1:]]
        assert meta["speaker_id"] not in interferer_spk           # target speaker never interferes
        assert len(set(interferer_spk)) == len(interferer_spk)    # distinct interferers
        assert meta["enroll_utt"] != meta["utt_id"]
        assert meta["enroll_utt"].split("-")[0] == meta["speaker_id"]
        assert item["speaker_label"] == data.spk2label[meta["speaker_id"]]


def test_dynamic_mix_speaker_count_distribution(fake_librimix) -> None:
    ds, _ = _dm_dataset(fake_librimix, n_spk_probs=[0.5, 0.3, 0.2])
    torch.manual_seed(0)
    counts = np.bincount([ds.num_speakers() for _ in range(4000)], minlength=5)[2:]
    assert np.allclose(counts / counts.sum(), [0.5, 0.3, 0.2], atol=0.03)


def test_dynamic_mix_relative_levels(fake_librimix) -> None:
    ds, _ = _dm_dataset(fake_librimix, n_spk_probs=[1.0], level_db_range=(0.0, 0.0),
                        partial_overlap_prob=0.0)
    torch.manual_seed(0)
    item = ds[0]
    t = item["target"].numpy()
    i = item["interference"].numpy()
    assert abs(20 * np.log10(active_rms(i) / active_rms(t))) < 0.1  # 0 dB when levels are fixed


def test_finetune_config_loads() -> None:
    cfg = resolve_paths(load_config(PROJECT_ROOT / "configs" / "finetune_multispk.yaml"))
    assert cfg["data"]["dynamic_mix"]["enabled"]
    assert cfg["train"]["init_from"].endswith("best.pt")
    assert [v["name"] for v in cfg["data"]["extra_val_sets"]] == ["3spk", "4spk"]
    assert all(v["root"].startswith(str(PROJECT_ROOT)) for v in cfg["data"]["extra_val_sets"])
