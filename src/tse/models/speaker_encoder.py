"""Speaker encoders: jointly trained ResNet (default) and an optional frozen ECAPA-TDNN."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GlobalLayerNorm, WaveEncoder


class ResBlock1d(nn.Module):
    """Residual 1D conv block (conv-BN-PReLU-conv-BN + shortcut, PReLU) followed by max-pool(3)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm1d(out_ch), nn.PReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm1d(out_ch),
        )
        self.short = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.act = nn.PReLU()
        self.pool = nn.MaxPool1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.body(x) + self.short(x)))


class ResNetSpeakerEncoder(nn.Module):
    """Enrollment waveform -> (shared) encoder -> gLN -> 1x1 -> 3 ResBlocks -> mean pool -> Linear.

    Args:
        encoder: The mixture encoder to share, or None to create a private one.
        N, L, stride: Encoder settings (used when ``encoder`` is None).
        channels: Channels of the 1x1 input conv and the three residual blocks,
            e.g. ``[256, 512, 512]`` (first entry is also the 1x1 output size).
        spk_dim: Output embedding size.
    """

    def __init__(self, encoder: WaveEncoder | None, N: int, L: int, stride: int,
                 channels: list[int], spk_dim: int) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else WaveEncoder(N, L, stride)
        c0 = channels[0]
        self.front = nn.Sequential(GlobalLayerNorm(N), nn.Conv1d(N, c0, 1))
        chans = [c0] + list(channels)
        self.blocks = nn.Sequential(*[ResBlock1d(chans[i], chans[i + 1]) for i in range(3)])
        self.out = nn.Linear(chans[3], spk_dim)

    def forward(self, enroll: torch.Tensor) -> torch.Tensor:
        """enroll: (B, Te) -> (B, spk_dim)."""
        h = self.blocks(self.front(self.encoder(enroll)))
        return self.out(h.mean(dim=-1))


class ECAPAFrozenSpeakerEncoder(nn.Module):
    """Frozen SpeechBrain ECAPA-TDNN (VoxCeleb, 16 kHz) + trainable Linear(192 -> spk_dim).

    Input is resampled from ``sample_rate`` to 16 kHz. Requires ``speechbrain``; the
    pretrained model is downloaded on first use into ``savedir`` (files are copied, not
    symlinked, so it works on Windows without developer mode). Only the feature extractor,
    input normalisation and embedding network are kept (not the VoxCeleb classifier). They
    stay in eval mode and run in float32 without gradients; only ``proj`` is trained.
    The embedding is L2-normalised and scaled to unit RMS per dimension before ``proj``.
    """

    EMB_DIM = 192

    def __init__(self, spk_dim: int, sample_rate: int = 8000,
                 source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 savedir: str | Path | None = None) -> None:
        super().__init__()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as err:
            raise ImportError(
                "spk_encoder='ecapa_frozen' needs speechbrain (pip install speechbrain). "
                "Use spk_encoder='resnet' otherwise.") from err
        import torchaudio

        if savedir is None:
            savedir = Path(__file__).resolve().parents[3] / "pretrained_models" / source.split("/")[-1]
        ecapa = EncoderClassifier.from_hparams(source=source, savedir=str(savedir),
                                               run_opts={"device": "cpu"},
                                               local_strategy=LocalStrategy.COPY)
        self.resample = torchaudio.transforms.Resample(sample_rate, 16000)
        self.compute_features = ecapa.mods.compute_features
        self.norm = ecapa.mods.mean_var_norm
        self.embedding_model = ecapa.mods.embedding_model
        for m in (self.compute_features, self.norm, self.embedding_model):
            for p in m.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(self.EMB_DIM, spk_dim)
        self.train(self.training)

    def train(self, mode: bool = True) -> "ECAPAFrozenSpeakerEncoder":
        """Only ``proj`` follows train/eval; the frozen ECAPA parts always stay in eval mode."""
        super().train(mode)
        for m in (self.compute_features, self.norm, self.embedding_model):
            m.eval()
        return self

    def ecapa_embedding(self, enroll: torch.Tensor) -> torch.Tensor:
        """Raw 192-d ECAPA embedding of (B, Te) waveforms at the model sample rate."""
        with torch.no_grad(), torch.autocast(enroll.device.type, enabled=False):
            wav = self.resample(enroll.float())
            feats = self.compute_features(wav)
            feats = self.norm(feats, torch.ones(wav.shape[0], device=wav.device))
            return self.embedding_model(feats).squeeze(1)

    def forward(self, enroll: torch.Tensor) -> torch.Tensor:
        """enroll: (B, Te) -> (B, spk_dim).

        The ECAPA embedding is L2-normalised (it is meant to be compared by cosine, and its raw
        norm of hundreds would destabilise FiLM), then scaled by sqrt(192) so every dimension
        has unit RMS. Without that scale the projected embedding starts about 4x smaller than
        the ResNet's. Because FiLM weights are zero-initialised and their gradient is
        proportional to the embedding, the speaker cue was then learned much more slowly.
        """
        e = F.normalize(self.ecapa_embedding(enroll), dim=-1) * self.EMB_DIM ** 0.5
        return self.proj(e)


def build_speaker_encoder(cfg: dict, mixture_encoder: WaveEncoder, sample_rate: int) -> nn.Module:
    """Build the speaker encoder selected by ``cfg['spk_encoder']``."""
    if cfg["spk_encoder"] == "resnet":
        shared = mixture_encoder if cfg["share_encoder"] else None
        return ResNetSpeakerEncoder(shared, cfg["N"], cfg["L"], cfg["stride"],
                                    cfg["spk_channels"], cfg["spk_dim"])
    if cfg["spk_encoder"] == "ecapa_frozen":
        return ECAPAFrozenSpeakerEncoder(cfg["spk_dim"], sample_rate)
    raise ValueError(f"Unknown speaker encoder {cfg['spk_encoder']}")
