# Qwen T/D/P Role-aware Training 设计

日期：2026-11-07 UTC。目标是在已经验证 Dynamic Binary 有效的 Qwen-Image-Edit-2511
CJT 接口上增加可学习的 T/D/P 区域监督，直接验证物体位移、阴影更新和背景保持是否改善。
本阶段不修改 `external/Inter-Edit`，也不使用封存 Test 做训练或调参。

## 2026-11-07 续跑修订（实施前登记）

`retry_04` 的实际失败是 CPU scheduler timestep 表被 CUDA indices 索引；新实现沿用固定
上游 `compute_density_for_timestep_sampling(weighting_scheme="none")` 的 CPU 采样，再将
timestep 搬到 GPU。FM 仍是 `noise - normalized_target`，权重仍为 1，不改变采样分布。

阴影方向改用有符号反事实响应
`delta = (target-target_no_shadow) - (source-source_no_shadow)`；只在 canonical valid D
区域内，按最大绝对值 RGB 分量的正/负分成 `D_remove/D_create`。这样覆盖重叠阴影的
强度变化，不把普通 RGB 差异作为因果标签。latent mask 用 FP32 面积平均保留有效覆盖率，
不经过 uint8 量化；ignore 不进入区域项。global FM 保留上游全网格均值作为统一基线。

两个方法消费同一 SHA 校验的 Train 编码缓存，初始化前独立重置 seed；记录 step/sample ID、
timestep、noisy/target/control/prompt 哈希、梯度范数、参数更新、峰值显存和逐步耗时。
保存初始化及每 20-step LoRA 和可恢复状态，120→240 延长须从原状态继续且保留原产物。
smoke 使用两个真实更新，并验证保存/重载与实际生成。

本轮先完整评估两个方法在 Train6 上的 20/40/60/80/100/120 checkpoint，固定推理20步；
成功后做 Train36 的同预算对照和 Dev12。所有 checkpoint 均保留，不能事后只挑最好的一张图。
若区域加权仍留下旧物体/旧阴影，将先用这些训练曲线和图片确定一个最小 effect-aware
候选，单独登记公式和预算后才执行；不预设其有效。

## 实验范围

实验固定 Qwen revision、Inter-Edit revision、synthetic 数据、Dynamic Binary control、
LoRA rank8、BF16、batch1、seed42 和采样协议。项目侧 trainer 的普通 FM 模式和 Role-aware
FM 模式使用完全相同的初始化、batch 顺序、噪声/timestep 生成和 checkpoint 预算；普通
Inter-Edit 20-step 结果继续作为历史锚点，不与新 trainer 的长训练结果混写。

第一阶段使用 6 个 Train 样本（两个 scene-group，包含 4 个 strength 0.5/1.0 非零编辑和
2 个 identity），每个方法训练 120 steps，每 20 steps 保存 checkpoint。它只回答“loss 和
生成是否能在少量样本上过拟合目标位移、旧位置清除及阴影变化”，不作为 Dev 质量结论。
第二阶段使用完整 36 个 Train
样本，每个方法训练 120 steps；只有 overfit 通过且 GPU/时间预算允许，才追加 240-step
扩展。普通 FM、Role-aware FM 先完成；Static/Shuffled 只在主对照显示需要时按相同预算
追加，不同时改变训练步数和模型。

## Role-aware FM Loss

训练监督仍然是 target latent 的 flow-matching velocity。每个 sample 的 supervision
只在训练 worker 内由 counterfactual evidence 和 role/valid 文件构造，Dynamic Binary
仍是唯一的模型控制图；target RGB 不进入 control image 或推理输入。

在 source/target occupancy 和阴影 intervention 上构造五个 latent-space region：

- `T_old = old \ new`，监督旧物体位置被清除；
- `T_new = new \ old`，监督新物体位置被生成；
- `D_remove`，反事实阴影减弱/移除的 valid D 区域，包含两侧都有阴影的强度变化；
- `D_create`，反事实阴影增强/创建的 valid D 区域；
- `P`，canonical valid 的 P 区域，保护未因果变化的背景。

region map 用 FP32 面积平均从 512² 对齐到实际 VAE latent H×W，invalid 像素对区域项为零。
latent 的分数覆盖是像素监督代理，不声称等价于 VAE 的非局部感受野。每个非空
region 先做像素数归一化，再对 active regions 等权平均，避免 P 的面积压过 T/D；少于 4
个 latent pixel 的 region 标记 inactive，不填入伪监督。最终损失固定为：

```text
global_fm = mean(all latent/channel errors)  # 保持官方普通 FM
role_fm = mean(equal_weighted(active region means))
loss = (1 - 0.5) * global_fm + 0.5 * role_fm
```

训练日志每步记录 global/role loss、region loss/coverage、sample/noisy/prompt SHA、梯度
和显存；checkpoint 在0和每20步保存。普通 FM 使用同一代码和相同 batch/noise，只将
`role_loss_weight=0`；这样
不会把 trainer 实现差异误当作方法收益。

## 验收与停止门槛

Overfit 必须同时满足：训练样本的 T/D target MAE 相对 source-only 下降、T_new 和
D_create 有明显变化、T_old source residual 下降、P drift 没有全图爆炸，并且至少一个
中间 checkpoint 的配对图可解释。只看 loss 下降不通过。

完整 Train 的主验收在 Dev，Train 只用于检查是否记忆训练样本。所有 checkpoint 使用固定
20-step inference 生成并运行已有 target-aware score，报告 T/D/P、T_old residual、
D_remove/D_create、shadow signed response/dark IoU、P edge drift、identity、changed
fraction、模板定位和失败案例。Role-aware FM 只有在 T_new/T_old、至少一个阴影方向和
P/identity 同时不劣于普通 FM 时，才登记为实际收益；否则保留诊断并从 loss 中撤销复杂化。

结果目录、配置快照、训练日志和失败回执全部放在 `outputs/`，Test 继续封存。任何 OOM、
GPU 争用或 supervision geometry 错误都单独记录，不覆盖历史 checkpoint。

资源边界：只有空闲卡才能通过 `image-edit gpu --` 的双次检查；worker 建立自身 CUDA context
后和每步训练前再次确认所租 UUID 只有一个 compute PID。若共享节点后来出现其他任务，只
退出本项目 worker，不中断其他 PID。推理也在建立 context 和逐样本前复核。

与官方的一致性指同一1024级预处理、posterior sample、mean/std、token packing、CPU均匀
timestep、velocity目标、LoRA target_modules和公开save/load接口。Q3显式固定缓存、初始化和
独立batch顺序，不声称与历史Q2的随机数消耗逐bit相同；新方法只在同一Q3实现内比较。

## 续训后的推理验收修订（实施前登记）

共享节点可能在已保存checkpoint后发生争用，首次execution失败、后续正常续训成功。
推理应接受同一run内成功的initial或continuation execution，而不是只看首次回执；失败或
running的回执不能单独授权推理。推理记录绑定所采用execution的路径与SHA，配对比较
再次核验，旧Q2缺少此字段时保留兼容。验收覆盖首次成功、失败后成功续训、只有失败/
running、returncode不一致和回执被修改；训练loss、数据、预算和模型参数均不改变。

## 2026-11-09 smoke 调度修订

首次真实 smoke 已完成六个样本的 VAE/text cache，并写出共享 `cache.pt`；缓存释放后，
项目没有保留设备分配，`nvidia-smi` 返回空 compute PID，旧的“必须恰好一个PID”检查把
安全状态误判为争用，0 steps更新后退出。修订后的门禁允许空PID集合（外层启动前已检查
显存/利用率且没有其他compute进程），若有PID则必须全部属于当前worker，任何其他PID或
多PID立即退出。该修复只改变资源门禁，不改变训练目标、缓存、随机数或模型控制；原失败
receipt保留，后续使用新run目录重跑。

这三次失败还显示共享节点的外部任务会在大型库导入期间占用刚释放的卡。训练入口现在
把第一次租约复核放在大型 diffusers/Qwen import 之前，后续仍在缓存、模型加载和每步前
复核；这是启动时序优化，不放宽外部PID拒绝规则。
