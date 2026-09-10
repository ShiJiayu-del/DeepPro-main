# BC-TPro Clean 训练实验的范围更正

2026-09-09 对 DeepPro 论文和作者官方代码复核后确认：论文表格中的
`NUDT-MIRSDT-HiNo` 主结果与核心结构消融使用的是在强噪声训练集上单独训练的
权重，而不是在 Clean 上训练后直接迁移到 HiNo。

因此，本目录已经完成的 12 个训练及其 Clean/Noise8 双条件结果继续保留，但其
定位更正为：

- Clean 域的受控结构消融；
- Clean 权重到 Noise8 的 zero-shot domain-shift 压力测试；
- 不能作为论文 HiNo 同域训练协议的主结果，也不能用于选择论文最终模型。

论文对齐的 Noise8 同域训练已迁移到相邻目录
`../bc_tpro_stage1_noise8_2026-09-09/`。两批实验的 checkpoint、metrics、分析和
结论必须分开报告，不得合并均值。此更正不删除或改写本目录已有结果。
