# BC-TPro Stage-1 CUDA incident record

## Symptom

The first formal Wave-1 attempt reached epoch 8 for all three baseline seeds,
then failed on the first full-resolution validation window. PyTorch reported
`CUDA error: an illegal memory access was encountered`. The kernel log recorded
one NVIDIA Xid 31 MMU fault on each of physical GPUs 0, 1, and 2 between
2026-09-08 23:44:00 and 23:44:10 CST.

The failed run directories, launcher logs, and status records are retained at:

`log/sem_seg/_queues/bc_tpro_stage1_2026-09-08/failed_attempts/2026-09-08_deterministic_conv3d_xid31_epoch8`

## Reproduction and isolation

With `CUDA_LAUNCH_BLOCKING=1`, the failing operation was localized to the
second `STD_Resblock` in `networks/layers/basic.py`: its
`SDifferenceConv` calls a 3-D convolution with 32 channels, kernel 3,
spatial dilation 2, and spatial padding 2. The first validation feature tensor
has shape `1×32×40×271×396`.

The failure is independent of validation AMP, `torch.inference_mode`,
DataLoader workers, the Soft-IoU loss, and TPro row chunking. A minimal
same-shape convolution reproduces the illegal access with deterministic cuDNN
enabled and succeeds with that path disabled. Patch training at 128×128 does
not select the failing full-resolution algorithm.

## Protocol-preserving workaround

Training remains deterministic and runs for the preregistered fixed 32 epochs.
In-process full-resolution validation is disabled with
`--skip_inprocess_validation 1`. The fixed `epoch_32_model.pth` is then
evaluated by fresh `test.py` processes. Full-frame evaluations are serialized
with a file lock.

A diagnostic epoch checkpoint followed by a fresh AMP evaluation completed all
16 validation sequences (48 windows), used about 1.50 GiB peak allocated CUDA
memory, and wrote a valid schema-v2 metrics JSON. This workaround does not
change model parameters, training updates, data, loss, or checkpoint selection.

## Separate first-attempt interruption

An earlier Wave-1 attempt was terminated when its foreground carrier session
ended near epoch 6. It was not resumed because the checkpoint format does not
capture all RNG states needed for an exactly paired restart. Those artifacts
are retained separately at:

`log/sem_seg/_queues/bc_tpro_stage1_2026-09-08/failed_attempts/2026-09-08_external_exec_sigterm_wave1_epoch6`
