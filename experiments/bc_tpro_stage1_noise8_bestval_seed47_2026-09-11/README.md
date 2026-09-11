# BC-TPro Noise8 seed47：逐 epoch 验证与 best checkpoint 重跑

## 纠正原因

此前 upstream FP32 与 nongate 结果在训练时设置了
`skip_inprocess_validation=1`，训练后固定评测 `epoch_32_model.pth`。这不符合本轮确认的
“每个 epoch 后验证、保存验证集最优 checkpoint、最终评测 best checkpoint”协议。旧训练
没有保存中间轮次权重，不能事后恢复真正的 best，因此旧指标只保留作历史证据，不再作为
当前模型结论。

## 本轮范围

只重跑当前有效的 Noise8、upstream-compatible、FP32、seed47 七个结构：B1、C0、C1、
C2、NG1、NG2、NG3。旧 Clean、AMP 和 seed49/51 结果不属于当前有效比较，不重复训练。

固定训练设置：scratch-only、32 epochs、batch4、T=40、Soft-IoU、train64/internal-val16，
每个 epoch 完整验证一次，关闭早停。`best_model.pth` 按 internal-val micro pixel IoU@0.5
最大化保存；精确相等时保留较晚 epoch。该指标只负责同一次训练内的 checkpoint 选择。
checkpoint 同时记录 `checkpoint_selection`（指标、方向、最佳值、最佳 epoch）和当前
`validation_metrics`，使后续评测能够核对实际加载的轮次。

训练完成后，独立 `test.py` 不传 `--epoch`，显式加载 `best_model.pth` 并在同一 internal-val16
上计算 Pd@0.5、Fa@0.5 和 AUC27。七个结构之间仍同时比较 Pd 越高、Fa 越低、AUC 越高，
不采用 AUC 优先或未登记加权分数。official test20 不参与逐轮验证或 checkpoint 选择。

## 隔离与资源

- 数据：`/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY`；
- 划分：复用
  [`../bc_tpro_stage1_noise8_upstream_2026-09-09/splits`](../bc_tpro_stage1_noise8_upstream_2026-09-09/splits)；
- 运行清单：[manifest.tsv](manifest.tsv)；机器协议：[PROTOCOL.json](PROTOCOL.json)；
- GPU：只使用物理卡 0、1、2，每卡同时最多一个训练；
- SwanLab：关闭，避免外部代理故障中断训练；本地日志与队列 marker 为权威证据；
- 新日志与队列使用 `2026-09-11` 和本实验名，不覆盖任何历史 checkpoint、日志或指标；
- 每轮 full-val 使用 `validation_safe_cudnn=1`，规避本机确定性全分辨率 cuDNN 路径的
  Xid 31；每次验证结束后立即恢复确定性训练设置。

## 状态

协议已登记，等待启动。预计七个任务三卡并行排队约 2 小时。完成前不得把旧 epoch32
结果或本轮未完成结果写成当前结论。

启动前只读检查：

```bash
/home/user/anaconda3/envs/sjyPID/bin/python tools/run_bc_tpro_bestval.py --dry-run
```

启动重跑：

```bash
/home/user/anaconda3/envs/sjyPID/bin/python tools/run_bc_tpro_bestval.py --run
```

七项完成后生成核验报告和 Excel：

```bash
/home/user/anaconda3/envs/compress/bin/python tools/analyze_bc_tpro_bestval.py --xlsx
```

训练固定使用 `sjyPID`；Excel 汇总使用已验证包含 `openpyxl` 的 `compress` 环境。
