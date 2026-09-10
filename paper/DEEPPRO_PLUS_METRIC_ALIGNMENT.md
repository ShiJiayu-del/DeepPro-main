# DeepPro-Plus 论文评价口径对齐

依据：Ruojing Li 等，*Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better*，
IEEE TPAMI 48(8):10157–10175，2026-08，DOI: 10.1109/TPAMI.2026.3683258。

## 论文主指标

NUDT-MIRSDT 与 NUDT-MIRSDT-HiNo 的主检测指标为：

- `Pd`（probability of detection）：检测到的真实目标数 / 真实目标总数；表中单位为百分数（10^-2）。
- `Fa`（false alarm rate）：目标保护区域以外的误警像素数 / 总像素数；表中单位为 10^-5。
- `AUC`：Pd-Fa ROC 曲线下面积，通过论文代码采用的预定义概率阈值组扫描得到。
- 效率指标：参数量、256x256 输入下每帧 GFLOPs、FPS。

深度学习方法的固定工作点是 sigmoid 后阈值 `0.5`。NUDT-MIRSDT 还单独报告
`SNR <= 3` 的 8 个测试序列：Sequence47、56、59、76、92、101、105、119。

当前 NUDT/Noise8 主结果、门控和模型排名只使用 Pd@0.5、Fa@0.5 和官方 27 阈值
Pd-Fa AUC。训练 loss/IoU 只作优化诊断；历史 payload 中的 Pixel Precision/Recall/F1
兼容字段不进入 active analyzer 或论文表。

## 论文 DeepPro-Plus 参照值

| Scenario | Pd | Fa (x10^-5) | AUC |
|---|---:|---:|---:|
| NUDT-MIRSDT, SNR <= 3 | 99.24 | 1.65 | - |
| NUDT-MIRSDT, all | 99.71 | 2.69 | 0.9978 |
| NUDT-MIRSDT-HiNo | 76.23 | 1.69 | 0.9171 |

官方 DeepPro-Plus 实际有 70,913 个标量参数，FP32 存储为 0.283652 MB（约 277 KiB）。
论文表中的 `0.284 M` 与存储 MB 对应，不能解释为 0.284 百万个参数。论文另报告
3.89 GFLOPs/frame（256×256）和 224.05 FPS（V100）；官方 README 的 480×720 口径与
当前 `test.py` 的 200×300 THOP 口径均不同，不能混用。不同 GPU 上的 FPS 还必须冻结
batch、预热和计时范围。

## 当前实验的可比性边界

当前 29 项从零训练实验使用同一数据、seed 和优化协议，因此适合比较结构增删的相对
影响。它们的检测评测将统一输出 Pd、Fa、AUC，并以本地 `deeppro_plus` 作为受控 baseline。

但是当前训练协议是 learning rate 0.005 与 `f1_calibrated_ohem`，而论文明确使用：

- 32 epochs，batch size 4；
- Adam，初始 learning rate 0.001；
- 每 10 epochs 将学习率乘 0.7；
- Kaiming convolution initialization；
- 40 帧输入、128x128 random crop；
- Soft-IoU loss；
- 推理窗口 10% 重叠并取 union/max 融合。

所以当前结果属于“评价指标对齐、训练协议受控但不同”，不能写成对论文数值的严格复现。
论文主表应同时列出论文已报告的 DeepPro-Plus 行和本地同协议 DeepPro-Plus 行，并清楚
标注来源。历史 29 模型已经评测过 official test20，因此它不是项目级从未见过的数据；
当前 BC-TPro Stage1 使用独立 internal-val16 锁定结构，但最终 test20 结果仍须披露历史
暴露。若需要声称在原论文训练协议下超过 baseline，还必须完成当前 upstream/FP32/
Soft-IoU/fixed-epoch 多 seed 协议，而不能复用旧单 seed/F1-OHEM 排名。
