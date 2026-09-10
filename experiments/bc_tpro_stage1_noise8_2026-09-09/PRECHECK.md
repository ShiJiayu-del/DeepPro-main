# Noise8 BC-TPro 启动前检查

检查时间：2026-09-09（Asia/Shanghai）。状态：**配置通过，GPU 训练未启动**。

## 已验证

- 数据根和 dataset name 精确为 Noise8.0_FJY 版本。
- 官方 `train.txt` 为 8,000 条、80 序列，每序列 100 条；官方 `test.txt`
  为 2,000 条、20 序列，每序列 100 条；两组序列无交集。
- 固定列表为 64 train / 16 internal val，无重复、无交集、并集严格等于官方
  train 80 序列；按 seed 20260908 复算 PCG64 选择一致。
- 64/16 列表与官方 test 20 序列无交集。测试列表只用于元数据隔离断言，未
  传入训练或验证命令。
- manifest 精确含 4 variants × 3 seeds；seed 47/49/51 分别固定 GPU 0/1/2；
  model、variant、run_id、wave、日期化 log_dir 均通过语义校验。
- dry-run 生成 12 条训练命令和 12 条 Noise8 internal-val 评测命令。
- 所有训练命令均为 epoch 32、batch 4、T=40、crop=128、Soft-IoU、SwanLab
  cloud、`resume=never`；三个预训练 checkpoint 参数均为空。
- `python -m py_compile tools/validate_bc_tpro_noise8_setup.py` 和
  `bash -n tools/run_bc_tpro_stage1_noise8.sh` 通过。
- 当前协议和 split manifest 不包含哈希字段，也不依赖 SHA256 前置检查。

## 正式运行前仍需现场确认

- GPU 0/1/2 无残留进程且各有足够显存。
- SwanLab 网络/代理可用；初始化失败时 launcher 会标记失败并停止后续 wave，
  不会静默离线训练或自动重试。
- `log/sem_seg/2026-09-09/...` 目标目录为空，专用 queue 无 `.failed` 状态。
