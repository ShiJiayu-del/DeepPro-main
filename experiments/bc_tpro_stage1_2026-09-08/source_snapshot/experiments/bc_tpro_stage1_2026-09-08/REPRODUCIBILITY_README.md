# BC-TPro Stage 1 复现冻结说明

`tools/snapshot_bc_tpro_stage1.py` 只能在 12 个训练 run 和 24 份
Clean/Noise8 评测 JSON 全部完成后执行。工具不读取图像内容、不创建
CUDA context、不修改训练产物，也不复制数据集或 checkpoint。

## 完成后执行

```bash
/home/user/anaconda3/envs/sjyPID/bin/python \
  tools/snapshot_bc_tpro_stage1.py
```

工具会先闭合校验：

- `manifest.tsv` 必须是 B1/C0/C1/C2 × seeds 47/49/51 的 12 个 run；
- 12 个 `.done` 标记的 run/seed/wave/GPU 必须与 manifest 一致；
- 必须恰有预期的 24 份 schema-2 评测 JSON，且数据集、序列顺序、
  40 帧设定和 epoch-32 checkpoint 引用一致；
- 每个 checkpoint 的 `model_name`/`structure_variant`/内部 epoch 必须
  与 manifest 一致；
- 12 个 run 自带的模型、adapter 和 loss 源码快照必须逐字节一致，
  且与 canonical 源码 SHA256 一致；
- 64/16 划分必须不重叠并完整分割 `train.txt`，Clean 与 Noise8
  的训练列表必须相同，`split_manifest.json` 内哈希必须匹配。

任一项不成立时以退出码 `2` 拒绝冻结，不生成看似完整但不可信的
manifest。

## 产物

- `source_snapshot/`：保持仓库相对路径的关键源码与实验协议快照；
- `SOURCE_SHA256SUMS`：`source_snapshot/` 内每个文件的 SHA256；
- `REPRODUCIBILITY_MANIFEST.json`：包含 12 个 checkpoint 和 24 份
  metrics 的路径/大小/SHA256，Git HEAD/branch/status/diff SHA，
  Python/依赖/PyTorch/CUDA/cuDNN/GPU/driver 信息，数据列表哈希和
  12-run 源码快照审计。

checkpoint 和 metrics 保留在原路径；manifest 只记录其引用和
SHA256。因此移动仓库后，可用 `repo_relative_path` 定位文件，同时用
`path` 保留原始机器上的证据位置。

## CPU 合同测试

```bash
python -m unittest tests.test_snapshot_bc_tpro_stage1
```

测试只使用临时小文件和 mock 环境，不读取真实 checkpoint，不调用
`nvidia-smi`，也不占用 GPU。
