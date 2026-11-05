# Qwen CJT 第二基座验证轨

2026-11-05 UTC。本文登记 Qwen 方向的研究设计和当前工程边界。模型下载、GPU pilot
和质量结论分别记录；任何 pilot 通过都不等于方法收益。

## 决策

增加 Qwen-Image-Edit-2511 作为第二基座，优先复用 Inter-Edit 官方 Control-Image
Joint Training（CJT）代码和预处理。选择 2511 是因为官方版本已经支持多图编辑，且
Inter-Edit 官方训练配方明确使用 `source image + binary edit mask` 两个对齐控制输入。
不直接切换到 Qwen-Image-2.0，也不把 Qwen 与 FLUX 的绝对分数做简单横向排名；这一轨
用于拆分“基座/注入接口失败”和“角色条件/标签失败”。

Qwen 官方架构使用 Qwen2.5-VL 语义编码、VAE 外观编码和 MMDiT，图像同时进入语义和
外观路径。Qwen-Image-Edit-2511 的模型权重规模为 20B，明显高于当前 4B FLUX；它
可以提供更强的编辑上界，但也会引入容量、预训练数据和架构的混杂因素。

## 必须回答的因果问题

1. Qwen 原生编辑能否在同一 synthetic source/instruction 上保持身份并完成位置变化。
2. 官方 binary-mask CJT 是否能复现稳定的 T/D/P 结果。
3. 在 Qwen 内，把 binary mask 替换为 role 或 geometry condition 是否带来额外收益。
4. 若 Qwen role 有效而 FLUX role 失败，问题是否主要位于 FLUX 的单点 residual 注入。
5. 若两个基座的 oracle role 都失败，是否应停止 T/D/P 表示并重做标签/任务定义。

## 实验矩阵

### Q0：不训练的基座能力检查

固定 Qwen-Image-Edit-2511 revision、source、instruction、scene split、sample ID 和
seed，比较 source-only、source+static mask、source+dynamic binary、source+role
RGB、source+geometry field。所有 oracle 条件只用于上界诊断；target RGB 不进入条件。

### Q1：官方 CJT 复现

先只做 `source + binary mask`，使用 Inter-Edit 的固定代码 revision、同一 source/target
配对、同一 prompt 和同一 train/dev/test 场景分割。只接受以下工程门槛后进入正式对照：

- source、mask、target 预处理和图像顺序逐项复现；
- 2-step/batch1 前向、反向、重载和有限梯度通过；
- 训练只读取 source/mask，target 只作为 flow target；
- 训练和推理的 mask 几何、VAE 预处理和控制数量完全一致；
- checkpoint、LoRA、数据和源码 revision 可重载。

### Q2：条件替换对照

保持 Q1 的 Qwen、LoRA rank、学习率、batch、更新数、source、target、prompt、seed 和
采样预算，只替换第二控制图：

|方法|第二控制图|目的|
|---|---|---|
|Static|source occupancy|无信息/静态控制基线|
|Dynamic Binary|T∪D|空间变化上界|
|Role RGB|T/D/P/ignore one-hot|复现当前角色上界|
|Geometry|old/new/dependency/direction/boundary|减少模型自行推断对应关系|

Role RGB 和 Geometry 都必须使用独立的 condition provenance。不得用目标 RGB、target
差分或精确 target footprint 作为推理输入。Geometry field 只作为 synthetic oracle
上界，未来 predictor 另行研究。

### Q3：只在 oracle 有收益后做 predictor

如果 Q2 的 role/geometry 在未见 scene 上稳定超过 Dynamic Binary，并且 P/identity 没有
恶化，才训练 predictor。predictor 输入限制为 source、instruction、粗 mask 和用户
strength；target、精确 new footprint、对象 ID、渲染参数只用于监督。predictor 需要
输出概率场和不确定性，低置信区域进入 ignore，而不是强行输出 P。

## 计算和资源边界

Qwen-Image-Edit-2511 的权重为 20B；官方 NVIDIA 架构说明还包括 7B Qwen2.5-VL 和
VAE，总组件规模约 27.1B。48GB A6000 不应假设能以完整 BF16 训练稳定运行。先做
单卡 capacity smoke：优先冻结 VLM、预缓存文本/图像 embedding，再评估 BF16 LoRA、
8-bit/4-bit quantized LoRA 和 CPU offload。若单卡无法在保留 OOM 余量下运行，使用两卡
模型并行或停止这条路线；不使用无效显存填充，也不在正式实验中临时降低分辨率或 batch
来改变预算定义。

官方 Inter-Edit debug 配置采用 batch1、1024 级 VAE 预处理、bf16、LoRA rank32、最多
两个 control images；本项目的 canonical MagicBrush 文件仍是 512²，但官方固定处理器会
把它映射到 1024 级 latent。当前 pilot 只做 2 个 Train 样本和 2 个更新，用来测量加载、
前向、反向、checkpoint 与重载边界，不把它当成 512² 质量实验。Qwen 方向的更新数与
FLUX 不能直接按数字等价，正式结论只在同一基座内部比较；跨基座只报告条件收益、质量
指标、显存、时间和预算。

2026-11-06 的实际 rank32 pilot 已完成加载、预缓存和第 1 个 update，第二个 update 在
LoRA backward 期间 OOM（峰值约 47.43/47.54GiB）。因此 rank32 只作为容量失败记录，
不进入正式条件比较；下一步先运行已登记的 `qwen_cjt_rank8_pilot.yaml`，只降低 LoRA
rank，其余预算和输入保持一致。若 rank8 仍无法保留显存余量，再评估 quantized LoRA
或双卡模型并行。

rank8 完成后必须先通过 CPU-offload 的单例 Dev checkpoint reload；该推理只接收 source、
binary mask 和 instruction，target 只在独立 provenance/evaluation 中使用。reload smoke
通过只证明 checkpoint 可消费和输入契约一致，不证明 Qwen 或条件方法有质量收益。

当前已实现 `src.qwen` 的 manifest bridge、model/external revision 检查、固定 seed 包装器、
leased runner 和 CPU-offload reload smoke。rank32 单卡容量失败，rank8 已完成 2-step
训练和 1 例 Dev reload；这些结果仍不构成质量结论。下一步才是固定 rank8 预算下的
Q1 binary-mask 对照，再决定是否进入 Q2 role/geometry 矩阵。
大权重下载遇到限流时只降低传输并发并保留断点分片；传输镜像不改变锁定 revision，
最终必须通过逐文件 SHA256 校验才会写入 `download.json`。

## 评估和停止条件

所有方法按 scene 等权，strength 先在 scene 内聚合。继续沿用 T/D target MAE、D 阴影
创建/移除、有向位移、P source drift、s=0 identity、对象定位、人工编辑成功和保留
评分。像素指标不能单独代替人工成功率。

任一条件满足以下情况即停止该方向：

- 官方 binary CJT 无法通过预处理/重载/有限梯度门槛；
- Qwen Dynamic Binary 和 Role/Geometry 在两 seed、Dev 上没有稳定差异；
- role/geometry 只改善像素指标却在 identity/P 或人工失败率上恶化；
- 只有 training scene 拟合，没有未见 scene 收益；
- 需要使用 target、精确 target mask 或事后调条件编码才能通过。

## 依据

- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen-Image technical report](https://arxiv.org/abs/2508.02324)
- [Inter-Edit official CJT repository](https://github.com/Delong-liu-bupt/Inter-Edit)
- [Inter-Edit debug configuration](https://raw.githubusercontent.com/Delong-liu-bupt/Inter-Edit/main/train_configs/train_config.yaml)
- [Inter-Edit full configuration](https://raw.githubusercontent.com/Delong-liu-bupt/Inter-Edit/main/train_configs/train_config_2511.yaml)
