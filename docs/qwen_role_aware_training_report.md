# Qwen Role-aware FM 训练阶段报告

日期：2026-11-07 UTC。当前报告只区分设计、代码验收和真实 GPU 运行状态；在成功完成
overfit 与 Dev 生成前，不把训练 loss 写成方法收益。

## 当前设计

训练固定 Qwen-Image-Edit-2511 revision、Inter-Edit revision、synthetic Dynamic Binary
control、rank8、BF16、seed42 和 batch1。普通 FM 与 Role-aware FM 使用项目侧同一个 trainer，
分别设置 `role_loss_weight=0` 和 `0.5`。Role-aware 监督在 VAE latent 网格均衡处理
`T_old/T_new`、`D_remove/D_create`、`P`，详细定义和停止门槛见
[训练设计](qwen_role_aware_training_design.md)。

第一阶段配置为 6 个 Train 样本、120 steps、每 20 steps checkpoint；第二阶段配置为完整
36 个 Train 样本、120 steps。只有 overfit 和资源余量都通过，才考虑 240-step 扩展。

## 已实现

- 新增 `src/qwen/role_loss.py`：证据驱动的五类 region、latent area resampling、面积归一化
  和 active-region equal weighting。
- 新增 `src/qwen/role_data.py`：复用 Qwen pipeline/VAE/text encoder 的 project-side cache，
  保留 source/control/target provenance。
- 新增 `src/qwen/role_train.py` 与 `qwen train-role`：普通 FM/Role-aware FM 共用 LoRA、
  noise、timestep、采样和 checkpoint 路径；训练循环精确跑满配置 steps，不受单 epoch 限制。
- inference/score 支持 `train` split，用于 overfit 记忆检查；Dev 仍是正式质量结论来源。
- 修复 Dynamic Binary full snapshot 与 train 子集 manifest 的合法 subset provenance。

## 已运行与失败记录

- CPU role-loss/bridge targeted tests：已通过；真实全量回归和 CI 在代码提交前执行。
- 1-step smoke 的三次启动均未形成质量结果：第一次是 JSON manifest/CSV 混读，第二次是
  Dynamic Binary 绝对 control path 解析，第三次在共享 GPU 竞态中 OOM。失败回执分别保留在
  `outputs/qwen_cjt_q3_smoke_role/`、`outputs/qwen_cjt_q3_smoke_role_retry_02/` 和
  `outputs/qwen_cjt_q3_smoke_role_retry_03/`；没有修改第三方源码，也没有中断他人进程。

正式 overfit、full Train、Train/Dev 生成和比较结果待稳定空闲 GPU smoke 通过后登记。
