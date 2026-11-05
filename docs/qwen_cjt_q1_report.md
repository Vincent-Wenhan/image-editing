# Qwen CJT Q1 实际报告

日期：2026-11-06 Asia/Shanghai。本文只报告 Qwen-Image-Edit-2511 的工程门禁和
有界 binary-mask 训练，不把单例生成或训练 loss 写成编辑质量结论。

## 冻结范围

- 模型：`Qwen/Qwen-Image-Edit-2511`，revision
  `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`。
- 外部训练器：Inter-Edit，revision
  `220c1dc26d8401cae160a6b25e48198f85bdd115`，仓库 clean。
- 数据：canonical MagicBrush 全量 Train/Dev，未读取 sealed Test；训练 200 条，Dev
  30 条，固定 `seed=42`、`rank=8`、`batch=1`、BF16、20 次 update、1024 级预处理。
- 条件：source image + binary edit mask，训练 target 只作为 flow supervision；推理只
  读取 source、mask 和 instruction。
- 协议：[`qwen_cjt_q1_binary_rank8.yaml`](../configs/experiments/qwen_cjt_q1_binary_rank8.yaml)。

## 实际结果

rank32 在 48GB 单卡的第二个 update 期间 OOM，峰值约 47.43/47.54 GiB，因此没有进入
条件比较。rank8 作为只改变 LoRA rank 的容量重试，73.73M 个可训练参数，2-step pilot
完成并保留了 checkpoint。

Q1 使用 rank8 完整训练成功：20/20 updates，执行时长 1474.77 秒，GPU UUID 为
`GPU-c805ad58-8e0a-13fc-c66b-08e2a98841fd`，观察到峰值约 48.6 GiB，无 OOM，最终
checkpoint 为 `checkpoint-20/pytorch_lora_weights.safetensors`，SHA256 为
`d6cf9c8f63ce94b77d145970cff8f09000583978ad1cc8be255a2f5089589a9f`。20 个 step loss
范围为 `0.00774–0.305`，均值约 `0.0490`；该波动只能说明这次短训练完成，不能证明泛化或
条件收益。

使用 checkpoint-20 做 CPU offload、4-step 的固定 Dev 单例重载成功，样本为
`dev_140513_t1`，预测 SHA256 为
`d35ecf5184061c7311a09e316973505503eb053df07b375145a16cb601362af4`。结果位于
`outputs/qwen_cjt_q1_binary_rank8/dev_inference_condition_safe/`。condition-only
校验在 target 文件被移除时仍可通过，推理记录保留 target SHA 作为 provenance，但没有
打开 target；这确认了输入/监督边界。

## 结论与限制

本轮工程门禁通过：模型资产、版本锁、manifest、固定 seed、GPU lease、rank8 训练、
checkpoint 和 target-free reload 均可复现。当前没有正式 T/D/P 指标、同预算 baseline、
第二个训练 seed 或真人评分，因此不能说 Qwen 或 CJT 已经改善项目方法。

Q1 使用 MagicBrush 的编辑 mask，只能验证通用编辑接口和训练链路，不能回答项目的
translation/direction/identity/geometry 因果问题。Qwen 的 20B 级基座也不能和 FLUX.2
Klein Base 4B 的绝对分数直接横向比较。

## 下一步决策

先停止扩大 MagicBrush 的 Qwen 训练，也不立即启动 role/geometry 大矩阵。下一项应准备
新的 synthetic scene 数据，使用独立 scene-group 的 Train/Dev/新封存 Test，并在 Qwen
内部按相同 checkpoint、rank、seed、update 和采样预算依次比较：

1. constant/no-information control，验证训练是否只是学会通用编辑；
2. dynamic binary `T∪D`，作为空间区域上界；
3. role 或 geometry condition，只在前两者链路稳定后进入。

这些条件必须共享 source/target/prompt/seed，target、精确新 footprint 和 role truth 只
用于训练监督或评估，不能进入普通推理条件。若 Qwen 的 binary 与 role/geometry 在新
scene 上没有稳定差异，停止继续增加条件编码；若只有 oracle 条件有效，则优先重做标签
可预测性和任务定义，而不是继续堆模型容量。

CPU 侧的下一轮资产已经固定：512²、20 个独立 scene-group，Train/Dev/Test 为 36/12/12，
并生成了 `constant_mask` 与 oracle `dynamic_binary` 两个 control snapshot。两个 Qwen
manifest run 已通过 full 和 condition-only provenance 校验；截至本报告更新时间，尚未
启动这两个 GPU 训练，因此没有把它们写成结果。

