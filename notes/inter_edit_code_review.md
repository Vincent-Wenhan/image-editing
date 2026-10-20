# Inter-Edit CJT 代码走读

核对版本：`220c1dc26d8401cae160a6b25e48198f85bdd115`；2026-10-20。
此处为源码分析，未运行 Qwen CJT 训练。

## 数据和空间条件

[`dataset_utils.py`](https://github.com/Delong-liu-bupt/Inter-Edit/blob/220c1dc26d8401cae160a6b25e48198f85bdd115/dataset_utils.py)
负责 JSON/JSONL/gzip 读取、路径和指令字段别名、`only_better_data` / `max_samples` 筛选。实际图像编码在训练脚本中，不能仅看这个文件就推断 mask 融合。

[`train_qwen_edit_lora.py`](https://github.com/Delong-liu-bupt/Inter-Edit/blob/220c1dc26d8401cae160a6b25e48198f85bdd115/train_qwen_edit_lora.py)
按 **[source, mask]** 顺序读取 RGB 控制图。约 997–1025 行：两图经条件图尺寸处理后，连同 instruction 进入 Qwen 的多模态 `encode_prompt`。约 1030–1057 行：target 和每张 control 分别用 VAE 编码。可预缓存文本/图像嵌入。

所以 mask 既参与多模态条件编码，也作为独立的 VAE 图像条件，不只是 loss mask、attention gate 或最终像素合成。公开接口称 binary mask，但读取时没有在每条路径重新阈值化；因此我们的数据实现必须独立约定插值、几何和取值，不能假设上游强制保证所有原始 mask 为二值。

## Flow matching 张量路径

训练脚本约 1122–1208 行：

1. target latent z 与 Gaussian ε 在扩散时间 σ 混合：zσ=(1−σ)z+σε。
2. pack 带噪 target；逐一 pack source/mask latents，控制图按同一样本组织。
3. 在图像 token 序列维连接 `[noisy_target, source, mask]`；`img_shapes` 分别描述各图形状。
4. transformer 同时接收图像 token、Qwen 条件嵌入、文本 attention mask 和 timestep。
5. 只截取 target token 对应的预测，unpack 后与 ε−z 计算 flow MSE（此路径 loss weighting 为 none）。没有以三角色作为逐区域监督损失。

[`pipeline_qwenimage_edit_plus.py`](https://github.com/Delong-liu-bupt/Inter-Edit/blob/220c1dc26d8401cae160a6b25e48198f85bdd115/pipeline_qwenimage_edit_plus.py)
的 `prepare_latents` 同样把多控制 latents 沿 token 维连接；采样时输入 noisy target 加 controls，再截取 target 预测。控制图次序与编码配置必须训练/推理一致。

## 移植到 FLUX 的边界

固定 ai-toolkit `a055947d56a156411abb4ef0301b35d0fe5f51a5` 的
`extensions_built_in/diffusion_models/flux2/flux2_model.py` 已声明
`has_multiple_control_images=True`、`use_raw_control_images=True`。可先给多个
`control_path` 实现 `[source, mask_rgb]` 的静态概念基线，但还必须核对位置 ID、VAE 缩放、batch 组织、训练采样和 Diffusers 多参考推理行为，并运行控制图置零/置换消融。

Klein 的 Qwen3 文本编码器并不是 Qwen-Image-Edit 的视觉语言编码器；不能复用 Qwen `encode_prompt(image=...)`，不能复用 Qwen latent packing/`img_shapes`，也不能把 CJT LoRA 权重直接加载进 FLUX。

建议先做相同接口/容量的 `[source, static mask]` 与 `[source, role RGB]`；若 VAE 对角色边缘或小区域表达不足，再开发空间编码器。先比较 oracle 和静态条件，避免在接口尚无作用时训练复杂 predictor。后续记录 role 通道语义、归一化、采样分辨率和强度编码；RGB 角色图本身不赋予模型因果理解。

## 评估可复用部分

官方 `evaluation/objective_metrics.py` 提供全局 LPIPS、区域 AlphaCLIP、Laplacian 边界平滑与参考图相似性。AlphaCLIP 需要额外权重和依赖；不是当前轻量诊断已实现的指标。VLM 评估还涉及外部推理配置，本次不运行。

MagicBrush 的 mask 外偏移只作为 proxy；它不等价于 P 区真值误差。角色阶段另需对象位移成功、阴影必要变化误差、P 区漂移、强度一致性和按场景的统计分析。公共 Inter-Edit benchmark 也不能直接提供强度条件角色真值。
