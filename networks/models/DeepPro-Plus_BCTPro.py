"""DeepPro-Plus with controlled raw-evidence BC-TPro ablations."""

import torch
import torch.nn as nn

from networks.layers.basic import SDifferenceConv, STD_Resblock
from networks.layers.TPro import TPro

try:
    # Training snapshots copy the exact adapter beside this model file.
    from bc_tpro_adapter import build_bc_tpro_adapter
except ImportError:
    from networks.layers.bc_tpro_adapter import build_bc_tpro_adapter


class detector(nn.Module):
    def __init__(
        self,
        num_classes,
        seqlen=100,
        out_len=100,
        structure_variant="none",
        structure_bottleneck_channels=8,
        eval_chunk_rows=0,
    ):
        super().__init__()
        self.out_len = out_len
        self.structure_variant = structure_variant
        self.eval_chunk_rows = int(eval_chunk_rows)
        if self.eval_chunk_rows < 0:
            raise ValueError("eval_chunk_rows must be non-negative")

        # This construction order exactly matches DeepPro-Plus.  The adapter
        # must not consume RNG before any common baseline module is initialized.
        self.conv_in = nn.Sequential(
            SDifferenceConv(
                in_channels=1,
                out_channels=8,
                kernel_size=(5, 7, 7),
                stride=(1, 1, 1),
                padding=(2, 3, 3),
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            STD_Resblock(8, 16),
            STD_Resblock(16, 32),
        )
        self.TPro = TPro(
            d_model=32,
            num_head=8,
            seqlen=seqlen,
            out_len=out_len,
        )
        self.conv_out1 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=8,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv_out2 = nn.Conv3d(
            in_channels=8,
            out_channels=num_classes,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )

        # Constructed last and applied after TPro, before the original head.
        self.bc_tpro = build_bc_tpro_adapter(
            structure_variant,
            channels=32,
            bottleneck_channels=structure_bottleneck_channels,
        )

    def _chunked_forward(self, features, raw_frames, return_aux):
        if return_aux:
            fused_evidence, auxiliary = self.bc_tpro.prepare_evidence(
                raw_frames, return_aux=True
            )
        else:
            fused_evidence = self.bc_tpro.prepare_evidence(raw_frames)
            auxiliary = None

        decoded_chunks = []
        residual_chunks = [] if return_aux else None
        for row_start in range(
            0, features.shape[1], self.eval_chunk_rows
        ):
            row_end = min(
                row_start + self.eval_chunk_rows,
                features.shape[1],
            )
            temporal_chunk = self.TPro(features[:, row_start:row_end])
            if fused_evidence is not None:
                evidence_chunk = fused_evidence[:, :, :, row_start:row_end]
            else:
                evidence_chunk = None
            if return_aux:
                temporal_chunk, residual_delta = (
                    self.bc_tpro.apply_fused_evidence(
                        temporal_chunk,
                        evidence_chunk,
                        return_delta=True,
                    )
                )
                if residual_delta is not None:
                    residual_chunks.append(residual_delta)
            else:
                temporal_chunk = self.bc_tpro.apply_fused_evidence(
                    temporal_chunk, evidence_chunk
                )
            decoded_chunks.append(self.conv_out1(temporal_chunk))

        decoded = torch.cat(decoded_chunks, dim=3)
        if return_aux:
            auxiliary["residual_delta"] = (
                torch.cat(residual_chunks, dim=3)
                if residual_chunks else None
            )
        return decoded, auxiliary

    def forward(self, seq_imgs, return_aux=False):
        seq_feats = self.conv_in(seq_imgs)
        seq_feats = self.layer1(seq_feats)
        seq_feats = seq_feats.permute(0, 3, 4, 1, 2)

        if (
            not self.training
            and self.eval_chunk_rows > 0
            and seq_feats.shape[1] > self.eval_chunk_rows
        ):
            # C0-C2 retain only the fused 8-channel raw evidence globally.
            # The baseline control retains the original low-memory path and
            # never assembles a full 32-channel post-TPro tensor.
            seq_feats, auxiliary = self._chunked_forward(
                seq_feats, seq_imgs, return_aux
            )
        else:
            seq_feats = self.TPro(seq_feats)
            if return_aux:
                seq_feats, auxiliary = self.bc_tpro(
                    seq_feats, seq_imgs, return_aux=True
                )
            else:
                seq_feats = self.bc_tpro(seq_feats, seq_imgs)
                auxiliary = None
            seq_feats = self.conv_out1(seq_feats)

        seq_midseg = self.conv_out2(seq_feats).squeeze(dim=1)
        if return_aux:
            return seq_feats, seq_midseg, auxiliary
        return seq_feats, seq_midseg


__all__ = ["detector"]
