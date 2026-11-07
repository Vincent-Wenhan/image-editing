# Qwen T/D/P Role-aware Training 设计

日期：2026-11-07 UTC。目标是在已经验证 Dynamic Binary 有效的 Qwen-Image-Edit-2511
CJT 接口上增加可学习的 T/D/P 区域监督，直接验证物体位移、阴影更新和背景保持是否改善。
本阶段不修改 `external/Inter-Edit`，也不使用封存 Test 做训练或调参。

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
- `D_remove`，source shadow effect 存在且 target shadow effect 消失的 D 区域；
- `D_create`，target shadow effect 存在且 source shadow effect 消失的 D 区域；
- `P`，canonical valid 的 P 区域，保护未因果变化的背景。

region map 按最近邻从 512² 对齐到实际 VAE latent H×W，invalid 像素完全排除。每个非空
region 先做像素数归一化，再对 active regions 等权平均，避免 P 的面积压过 T/D；少于 4
个 latent pixel 的 region 标记 inactive，不填入伪监督。最终损失固定为：

```text
global_fm = mean(all valid latent/channel errors)
role_fm = mean(equal_weighted(active region means))
loss = (1 - 0.5) * global_fm + 0.5 * role_fm
```

训练日志同时记录 global loss、role loss、五个 region loss、active pixel/count 和每步
checkpoint。普通 FM 使用同一代码和相同 batch/noise，只将 `role_loss_weight=0`；这样
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
