# Qwen CJT Q2 条件对照实际报告

日期：2026-11-06 UTC。本文区分训练门禁、正式 Dev 代理评估和研究决策；像素、模板和
阴影指标都不是真人评价，也不代表真实照片上的语义成功率。

## 固定协议

- 数据：`qwen_cjt_synthetic_20261106_retry`，512²，20 个独立 scene-group，Train/Dev/Test
  为 36/12/12；Test 没有进入训练 manifest。
- 模型和外部 revision 与 Q1 相同；Qwen LoRA `rank=8`、`seed=42`、BF16、batch1、20
  updates、1024 级预处理。
- 三个 control snapshot 使用相同 source/target/prompt/seed，只替换第二控制图：
  `constant_mask`、canonical `static_mask` 和 oracle `dynamic_binary`。
- 固定协议：[constant](../configs/experiments/qwen_cjt_q2_synthetic_constant_rank8.yaml)、
  [static](../configs/experiments/qwen_cjt_q2_synthetic_static_rank8.yaml)、
  [dynamic](../configs/experiments/qwen_cjt_q2_synthetic_dynamic_rank8.yaml)。

正式三方评估固定为 `seed=42`、20 sampling steps、全部 12 个 Dev 样本、4 个
scene-group，并使用同一 source、prompt、输出尺寸和 target-free inference 流程。未微调
Qwen 的 source-only baseline 使用同一 run 的数据与模型 receipt，但不加载 LoRA，只把
source 送入 pipeline；Constant 和 Dynamic Binary 都重载各自已完成的 `checkpoint-20`。
target、role、valid 和 counterfactual evidence 只在独立 score/gallery 阶段读取。

## 训练与失败记录

第一次 Constant 启动在 GPU 状态竞态下被用户要求停止，保留在
`outputs/qwen_cjt_q2_synthetic_constant_rank8/execution.json`，状态为
`failed/KeyboardInterrupt`，没有 checkpoint。

随后重新 prepare 到 `retry_02`，实时 UUID lease 确认 GPU0 无 compute PID 后启动，训练
20/20 updates 完成，执行时长 1197.59 秒，无 OOM，观察峰值约 48.5 GiB。20 个 step loss
范围为 `0.000146–0.036`，均值约 `0.003137`。最终 checkpoint 为
`outputs/qwen_cjt_q2_synthetic_constant_rank8_retry_02/checkpoints/checkpoint-20/`，
SHA256 为
`0d8810c6969169229eb38d47af762eddc20eccc2b8f474ccd20734fe285ce411`。

首次 Constant formal-20 reload 因共享 GPU 在模型加载期间重新出现 compute 进程而 OOM，
回执保留在
`outputs/qwen_cjt_q2_synthetic_constant_rank8_retry_02/dev_inference_formal20/inference.json`。
随后在独立 UUID lease 上使用 `dev_inference_formal20_retry_02` 完成全部 12 例；这两次
结果没有混合。

canonical static mask baseline 已登记并完成 manifest/control provenance，但首次 GPU 尝试
没有形成 checkpoint：`outputs/qwen_cjt_q2_synthetic_static_rank8/execution.json` 为
failed，模型搬到 GPU 时另一组 compute 进程占用约15.7 GiB。该回执只说明共享状态竞态，
不解释为 static 方法容量失败；本轮没有绕过门禁重试。

oracle Dynamic Binary 的 36/12 manifest、control SHA、source pairing 和 target-free
校验均已通过。20/20 updates 在 1228.19 秒内完成，无 OOM，观察峰值约 48.5 GiB；20 个
step loss 范围为 `0.000152–0.0395`，均值约 `0.003221`。checkpoint-20 SHA256 为
`44d569e93423c2a8fa29bef276e953a84d3d7849d0ca23355be04d3f80f028e7`，CPU
`qwen verify --require-model` 通过。随后使用 GPU4 UUID lease 完成全部 12 例 formal-20。

## Formal-20 Dev 结果

三方都完成 12/12 prediction，pipeline 原生输出统一为1024²后按 source 尺寸保存512²。
正式 score 目录为：

- source-only：`outputs/qwen_cjt_q2_formal_source_only/score_formal20/metrics.json`；
- Constant：`outputs/qwen_cjt_q2_synthetic_constant_rank8_retry_02/dev_score_formal20_retry_02/metrics.json`；
- Dynamic Binary：`outputs/qwen_cjt_q2_synthetic_dynamic_rank8/dev_score_formal20_retry_02/metrics.json`。

nonzero Dev 的 scene 等权聚合如下，MAE 和 drift 越低越好；`changed fraction` 只表示
像素变化代理：

|condition|T target MAE|D target MAE|P source drift|target MAE|changed fraction|identity target MAE|
|---|---:|---:|---:|---:|---:|---:|
|source-only Qwen|0.06748|0.16555|0.02803|0.03292|0.22109|0.00697|
|Constant|0.14372|0.21852|0.06577|0.07515|0.53188|0.00733|
|Dynamic Binary|0.05591|0.08773|0.03890|0.04057|0.20605|0.00583|

正式三方比较和 2000 次 scene bootstrap 回执在
`outputs/qwen_cjt_q2_formal_comparison_20/comparison.json`。Constant 减 Dynamic 的
差值为：T `0.08782 [0.05075, 0.12488]`、D `0.13078 [0.08453, 0.17572]`、P
`0.02687 [-0.02836, 0.08793]`、target MAE `0.03458 [-0.01545, 0.08923]`；changed
fraction 为 `0.32582 [-0.02785, 0.67102]`。因此 Dynamic 在 T/D 上的优势在本轮配对
scene 中稳定，P 和全图 MAE 的区间包含零，整体变化面积也没有稳定差异。

与 source-only 比较时，source-only 减 Dynamic 的 T/D 差值分别为
`0.01157 [0.00358, 0.01934]` 和 `0.07782 [0.05054, 0.10511]`，说明 Dynamic 对目标
位移和 D 区域更有效；P 为 `-0.01087 [-0.07946, 0.03809]`，target MAE 为
`-0.00765 [-0.06709, 0.03518]`，两者都不能证明 Dynamic 在全图或保护区更优。

模板/阴影代理进一步给出 nonzero Dev 的 scene 等权结果。目标 occupancy 的平均位移为
`96.93 px`；三种方法的模板定位都只有 `4/8` 个 nonzero 样本成功，其余显式记录为
abstain：

|condition|position error (px)|distance to old (px)|D shadow removal MAE|D shadow creation MAE|D signed response|D dark IoU|P edge MAE|
|---|---:|---:|---:|---:|---:|---:|---:|
|source-only Qwen|52.01|146.87|0.03294|0.18496|0.49592|0.31792|0.00173|
|Constant|69.86|74.68|0.30726|0.20656|0.33178|0.29735|0.00138|
|Dynamic Binary|34.57|64.08|0.32280|0.04054|0.73733|0.71597|0.00112|

这些代理显示 Dynamic 能更稳定地响应新目标/阴影方向，尤其是 shadow creation、signed
change 和 dark-region IoU；shadow removal 代理没有改善，模板定位的 abstain 也限制了
位移结论。它们用于定位失败模式，不是物理真值或真人评分。

## 配对图、失败样本和研究决策

全部 Dev 的配对图、四个最高 target MAE 失败案例和选择记录在
`outputs/qwen_cjt_q2_formal_gallery_20/paired_dev.png`、`failure_cases.png` 和
`failures.json`。失败记录保留了同一 sample 的三方 target MAE，能看到 Constant 在部分
样本产生大范围场景漂移，Dynamic 在一个强位移样本也可能复制目标物体；这些图不计作真人
评价。

结论限定为本 synthetic 数据、单训练 seed、4 个 scene-group、rank8/20 updates：

1. **Qwen 具备基础位移/阴影编辑能力。** source-only 已产生有限变化；加入 oracle
   Dynamic Binary 后，T/D 误差、目标定位代理和主要阴影创建代理明显改善。这证明了 Qwen
   基座在这个可控编辑任务上存在基础可学习性，但不能外推到真实照片或复杂语义编辑。
2. **Dynamic Binary 相对 Constant 带来有限而可靠的局部收益。** T、D、shadow creation
   和 dark-region IoU 的配对证据支持收益；P、全图 MAE、changed fraction 的收益尚未
   稳定。Dynamic Binary 使用 counterfactual geometry 的 oracle control，不能当成可部署
   条件或角色预测器结果。
3. **值得进入受限的 T/D/P Role-aware Supervision 阶段，但不扩大训练。** 下一步只登记
   一个同数据、同 rank8、同 seed、同 20-update 预算的三色 T/D/P role-map oracle
   条件，与本轮 Dynamic Binary 和 source-only 共用 Dev；预先设定 T/D、shadow creation、
   P drift 和 identity 四项门槛，并加入 role shuffle negative。若 role-map 没有同时
   保住 P/identity 且改善 T/D，就停止进入 predictor/更大预算；本轮不启动这项训练。

本轮没有使用双卡 DDP：Inter-Edit 当前 Accelerate 配置是 `num_processes=1`，普通 DDP
会在每张 GPU 复制完整 20B 模型，不能降低单卡显存峰值，还会改变有效 batch/更新语义。
若后续确需双卡模型并行，应单独设计和验证 lease、分片加载及预算，不把普通 DDP 当成显存
优化。static mask baseline 仍只完成登记，首次训练失败是共享 GPU 竞态；它不进入本轮结论，
也不因此追加训练。
