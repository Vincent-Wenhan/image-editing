# Qwen CJT Q2 条件对照实际报告

日期：2026-11-06 Asia/Shanghai。本文只记录 synthetic control 的工程训练门禁；没有
把 constant baseline 的训练 loss 写成编辑质量结论。

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

## Constant 结果

第一次启动在 GPU 状态竞态下被用户要求停止，保留在
`outputs/qwen_cjt_q2_synthetic_constant_rank8/execution.json`，状态为
`failed/KeyboardInterrupt`，没有 checkpoint。

随后重新 prepare 到 `retry_02`，实时 UUID lease 确认 GPU0 无 compute PID 后启动，训练
20/20 updates 完成，执行时长 1197.59 秒，无 OOM，观察峰值约 48.5 GiB。20 个 step loss
范围为 `0.000146–0.036`，均值约 `0.003137`。最终 checkpoint 为
`outputs/qwen_cjt_q2_synthetic_constant_rank8_retry_02/checkpoints/checkpoint-20/`，
SHA256 为
`0d8810c6969169229eb38d47af762eddc20eccc2b8f474ccd20734fe285ce411`。

训练完成后 GPU0 lease 已释放，CPU `qwen verify --require-model` 通过。constant 的
checkpoint reload、完整 Dev 生成和 score 结果与 dynamic 使用相同的采样预算。

## Dynamic binary 结果与决策

oracle dynamic binary 的 36/12 manifest、control SHA、source pairing 和 target-free
校验均已通过。随后在另一轮实时 UUID lease 确认 GPU0 无 compute PID 后完成训练：20/20
updates，1228.19 秒，无 OOM，观察峰值约 48.5 GiB。20 个 step loss 范围为
`0.000152–0.0395`，均值约 `0.003221`。checkpoint-20 SHA256 为
`44d569e93423c2a8fa29bef276e953a84d3d7849d0ca23355be04d3f80f028e7`，CPU
`qwen verify --require-model` 通过。

constant 和 dynamic 的训练门禁都已通过。随后使用相同 `dev_20261106_000_s000`、seed42、
4 steps、CPU offload 完成两组 checkpoint reload。Qwen pipeline 原生输出为1024²，推理
记录明确保存 `native_size=[1024,1024]`，再按 source 尺寸保存512² prediction；两组都记录
`resized_to_source=true`，避免把1024² smoke 直接混入512² synthetic评估。

- constant prediction SHA256：
  `4c3c3bc9c0f4a1400a82eef5129c359d933f727a3b868d6266da93d746d4d99c`。
- dynamic prediction SHA256：
  `ee15aacc3a01b8f9cfb3a9a6d9bfcee80a47edfd0850a5e4feca4a365c1b982d`。

该 identity 样本的 T/D 区域为空；constant 的 P source drift/target MAE 为 `0.02415`，
dynamic 为 `0.03034`，只作为单例尺寸和评估链路 smoke。

## 完整 Dev paired score

两组都在固定 12 个 Dev 样本、4 个 scene-group、seed42、4 sampling steps 下生成，并使用
独立 CPU score 读取 target/role/valid。nonzero Dev 的 scene 等权聚合如下，数值越低越好：

|condition|T target MAE|D target MAE|P source drift|target MAE|changed fraction|
|---|---:|---:|---:|---:|---:|
|constant|0.17394|0.15552|0.12892|0.13251|0.85950|
|dynamic binary|0.10595|0.07021|0.09215|0.09242|0.57598|

配对方向为 `dynamic - constant`，按 4 个 scene-group bootstrap 2000 次：T 为
`-0.06798 [-0.07640,-0.05748]`，D 为 `-0.08531 [-0.11886,-0.05908]`，P 为
`-0.03678 [-0.07289,-0.00066]`，target MAE 为 `-0.04009 [-0.07183,-0.00834]`。
changed fraction 的差为 `-0.28353 [-0.60082,0.01449]`，说明 dynamic 输出改变的
区域更少；这可能是更少 source drift，也可能是更保守的编辑，不能单靠这些像素指标判定
编辑成功。identity strength=0 的 target MAE 为 constant `0.03196`、dynamic `0.04117`，
dynamic 的无编辑保持还有退化信号。

完整回执保存在 `outputs/qwen_cjt_q2_comparison_v3/comparison.json`；两组逐例 metrics
分别在 `dev_score_all_v3/metrics.json`。这些结果只支持“oracle dynamic binary 在本轮
有限预算上比 constant 更接近 synthetic target 区域”，不支持部署条件收益、语义成功或
角色编码有效。

本轮没有使用双卡 DDP：Inter-Edit 当前 Accelerate 配置是 `num_processes=1`，普通 DDP
会在每张 GPU 复制完整 20B 模型，不能降低单卡显存峰值，还会改变有效 batch/更新语义。
若后续确需双卡模型并行，应单独设计和验证 lease、分片加载及预算，不把普通 DDP 当成
显存优化。static mask baseline 已登记，下一步先完成 static 与第二训练 seed，检查 dynamic 的
T/D收益是否在 identity/changed fraction 门槛下仍成立；之后才考虑 role/geometry oracle，
最后才是 predictor。
