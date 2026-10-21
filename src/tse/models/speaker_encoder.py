"""Speaker encoders: jointly trained ResNet (default) and an optional frozen ECAPA-TDNN."""

from __future__ import annotations

import torch
import torch.nn as nn

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

    Input is resampled from ``sample_rate`` to 16 kHz. Requires ``speechbrain`` and
    access to the pretrained model (downloaded on first use).
    """

    def __init__(self, spk_dim: int, sample_rate: int = 8000,
                 source: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        super().__init__()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as err:
            raise ImportError(
                "spk_encoder='ecapa_frozen' needs speechbrain (pip install speechbrain). "
                "Use spk_encoder='resnet' otherwise.") from err
        import torchaudio

        self.resample = torchaudio.transforms.Resample(sample_rate, 16000)
        ecapa = EncoderClassifier.from_hparams(source=source, run_opts={"device": "cpu"})
        self.ecapa = ecapa.mods
        self.compute_features = ecapa.mods.compute_features
        self.norm = ecapa.mods.mean_var_norm
        for p in self.ecapa.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(192, spk_dim)

    def forward(self, enroll: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.ecapa.eval()
            wav = self.resample(enroll.float())
            feats = self.compute_features(wav)
            feats = self.norm(feats, torch.ones(wav.shape[0], device=wav.device))
            emb = self.ecapa.embedding_model(feats).squeeze(1)
        return self.proj(emb)


def build_speaker_encoder(cfg: dict, mixture_encoder: WaveEncoder, sample_rate: int) -> nn.Module:
    """Build the speaker encoder selected by ``cfg['spk_encoder']``."""
    if cfg["spk_encoder"] == "resnet":
        shared = mixture_encoder if cfg["share_encoder"] else None
        return ResNetSpeakerEncoder(shared, cfg["N"], cfg["L"], cfg["stride"],
                                    cfg["spk_channels"], cfg["spk_dim"])
    if cfg["spk_encoder"] == "ecapa_frozen":
        return ECAPAFrozenSpeakerEncoder(cfg["spk_dim"], sample_rate)
    raise ValueError(f"Unknown speaker encoder {cfg['spk_encoder']}")
