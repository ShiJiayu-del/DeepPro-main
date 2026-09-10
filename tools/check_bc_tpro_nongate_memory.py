"""Measure two FP32 optimizer steps on one real train64 batch; never save weights."""
import gc
import importlib
import json
import os
from pathlib import Path
import random
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# This smoke check owns physical GPU0 only. The main queue is launched afterwards.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import torch
from data_utils.TrainDataLoader import TrainIRSeqDataLoader
from networks.losses.segmentation_losses import LegacySoftIoULoss


def main():
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = REPO.parent / 'datasets' / 'NUDT-MIRSDT-Noise8.0_FJY'
    experiment = REPO / 'experiments' / 'bc_tpro_nongate_noise8_seed47_2026-09-10'
    split = REPO / 'experiments' / 'bc_tpro_stage1_noise8_upstream_2026-09-09' / 'splits'
    dataset = TrainIRSeqDataLoader(
        'NUDT-MIRSDT-Noise8.0_FJY', data_root=str(data), seq_len=40,
        sample_rate=0.1, patch_size=128, sequence_list_file=str(split / 'train_sequences.txt'),
        upstream_compat=True,
    )
    raw, target = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)))
    assert tuple(raw.shape) == (4, 1, 40, 128, 128), raw.shape
    raw, target = raw.float().cuda(), target.float().cuda()
    model_type = importlib.import_module('networks.models.DeepPro-Plus_BCTPro').detector
    result = {'scope': 'smoke_only_no_saved_weights', 'dataset_length': len(dataset),
              'batch_shape': list(raw.shape), 'physical_gpu': 0, 'runs': []}
    for variant in ('center_multiscale', 'center_ring_difference', 'temporal_bandpass', 'center_spatial_smooth'):
        torch.manual_seed(47)
        model = model_type(1, 40, 40, structure_variant=variant, eval_chunk_rows=32).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = LegacySoftIoULoss()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        gradients = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(raw)
            loss = criterion(logits, target)
            assert torch.isfinite(loss), (variant, step)
            loss.backward()
            upstream_gradient = model.bc_tpro.evidence_fusion[0].weight.grad
            projection_gradient = model.bc_tpro.residual_projection.weight.grad
            assert torch.isfinite(projection_gradient).all()
            if step == 1:
                assert upstream_gradient.abs().sum().item() > 0, variant
            gradients.append({'fusion_l1': upstream_gradient.abs().sum().item(),
                              'projection_l1': projection_gradient.abs().sum().item()})
            losses.append(loss.item())
            optimizer.step()
            del logits, loss
        torch.cuda.synchronize()
        row = {'variant': variant, 'parameters': sum(p.numel() for p in model.parameters()),
               'losses': losses, 'gradients': gradients,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
        assert row['parameters'] == 71233
        result['runs'].append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer, criterion, upstream_gradient, projection_gradient
        gc.collect()
        torch.cuda.empty_cache()
    output = experiment / 'MEMORY_SMOKE.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('Wrote', output)


if __name__ == '__main__':
    main()
