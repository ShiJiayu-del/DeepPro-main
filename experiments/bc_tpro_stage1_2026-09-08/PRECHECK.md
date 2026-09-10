# BC-TPro 第一阶段启动前检查

检查时间：2026-09-08（Asia/Shanghai）。

## 通过项

- GPU 0/1/2 均为空闲 RTX 3090 24 GiB；GPU 3 被运行策略排除。
- 数据实际路径为 `../datasets/NUDT-MIRSDT` 和 `../datasets/NUDT-MIRSDT-Noise8.0_FJY`，未使用已经不存在的 `datasets_v1`。
- 固定划分为 train 64、val 16，二者交集为 0、并集严格等于官方 `train.txt` 的 80 个序列，与官方 `test.txt` 交集为 0。
- source/train/val SHA256 分别为：
  - `31e71368775adbc6c893bcb29ae1f39df6ffc83a693f42f84a4322bcf5235a8c`
  - `3e18da9a5c8dccea57b5155c244de1327ded05c6867f0059336f7a231ebbd967`
  - `bb92ecfdb0c379acc9971eaffed99154a2f985b063fd1f091aa8bceb09b07338`
- 11/11 BC-TPro CPU 单元测试通过：原始输入证据、零 DC/单位 L2、环形边界、时间反转、同 seed 公共初始化、初始 logits、两步梯度和 full/chunk 一致性均已覆盖。
- `py_compile`、`bash -n`、`git diff --check` 均通过。
- 启动器 dry-run 精确生成 12 条训练命令和 24 条 Clean/Noise8 验证命令。
- 所有训练命令的 `base_ckpt/spatial_ckpt/st_ckpt` 均为空，且 `train.py` 具有 scratch-only 硬拒绝逻辑。
- 真实 batch=4、40×128×128、C2、AMP 的一次前向/反向/Adam 更新通过：loss `0.99689382`，峰值 allocated `3.716 GiB`、reserved `4.289 GiB`。

## 参数量

| Variant | 总参数 | 新增参数 |
|---|---:|---:|
| B1 `none` | 70,913 | 0 |
| C0 `temporal_control` | 71,233 | 320 |
| C1 `center_multiscale` | 71,233 | 320 |
| C2 `center_ring` | 71,281 | 368 |

C0/C1 参数严格相同；C2 相对 C0/C1 只多 48 个静态融合参数。

## 环境与代码状态

- Python 3.8.5
- PyTorch 2.1.2 + CUDA 12.1
- SwanLab 0.7.15，正式实验使用 cloud 模式
- Git branch：`nudt-mirsdt-all-models-2026-09-01`
- 启动前 HEAD：`2914290fe8ee837855bc31d3e26dfac829954226`

工作树在本实验开始前已有大量未提交的历史整理修改，因此 HEAD 不能单独代表本次源码。每个运行目录会保存模型、BC-TPro adapter、loss 源码快照，checkpoint 保存完整参数和构造配置；本轮新增代码不得与历史修改混合宣称为已经推送的提交。
