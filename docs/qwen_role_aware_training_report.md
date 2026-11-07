# Qwen Role-aware FM 训练阶段报告

日期：2026-11-07 UTC。当前报告只区分设计、代码验收和真实 GPU 运行状态；在成功完成
overfit 与 Dev 生成前，不把训练 loss 写成方法收益。

14:11 UTC续跑：用户确认没有预约资源，仍按空闲门禁继续。复查8卡均有其他compute进程。
另修复续训验收：首次execution失败后，同run成功的continuation可供推理使用，绑定其
execution SHA并由comparison复核；只有失败/running或returncode非零的回执仍拒绝。
历史Q2回执保持兼容，不修改任何旧execution文件。验收与下一次资源等待另行登记于
`outputs/qwen_cjt_q3_20261107_v2/continuation_validation_02/` 和 `smoke_wait_02/`。
续训回执、比较与bridge的32项针对性测试全部通过（99.47s），Ruff check/format通过；
训练目标、共享编码缓存policy和checkpoint数值路径均未改变，未重跑已通过的昂贵CPU模型检查。

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

续跑核实：前一轮等待器已结束，`outputs/qwen_cjt_q3_smoke_role_retry_04/` 实际为 failed，
异常是 CUDA indices 索引 CPU scheduler timestep 表；没有完成训练更新。以下修复不覆盖
四个旧 smoke receipt。

2026-11-07 当前修复：沿用上游 CPU uniform timestep density / unit weighting 和 token
packing；五组监督改用 signed counterfactual D 与 FP32 latent area pooling。共享编码cache、
初始化checkpoint、batch schedule、逐步输入SHA/梯度/显存、LoRA公开save/load和optimizer/
RNG continuation均已实现。旧单epoch/字节mask/无输入审计的路径已被替代。

已运行的CPU验收包含小型真实Qwen Transformer的forward/backward/LoRA update，以及
公开pipeline loader的reload；重载输出逐元素相等。这是数值与接口契约，不是20B GPU
smoke或编辑质量结果。首批25项和后续10项targeted tests通过（两批存在重复项）。

对实际Train6的supervision审计位于
`outputs/qwen_cjt_q3_20261107_v2/supervision_train6/audit.json`：实际网格128²，valid D
遗漏为0；两个strength=1样本的D_remove有效latent面积为91.8125、116.4375，D_create为
141.125、187.25。strength=0.5没有独立D_remove区域；T_old仍单独监督旧位置清除。
这6个硬阴影场景没有有效重叠阴影强度变化像素，新定义的该能力只通过专门CPU回归证明。

续跑的机器可读预算在 `outputs/qwen_cjt_q3_20261107_v2/plan.json`；两方法的Train JSON字节
SHA相同，6例/两个scene-group、rank8/batch1/seed42/lr=2e-4/120 updates/每20步保存。
`smoke_01` 为独立两更新工程检查，`overfit_ordinary` / `overfit_role` 为正式候选；当前不能
把 prepared 状态写成已经训练。

2026-11-07 资源状态：12:16:50–12:47:29 UTC的有界等待已结束，22次GPU租约申请均因
没有空闲卡被拒绝，最终状态为 `no_idle_gpu`，没有启动模型worker。8张卡均有compute
进程；低利用率或只占264MiB的进程也不视为空闲。每次拒绝和最终状态保存在
`outputs/qwen_cjt_q3_20261107_v2/smoke_wait/`，等待器已退出。因此本轮还没有新的20B
更新、生成图、120-step曲线或
Role-aware收益结论；这是资源阻塞，不是方法/训练预算的失败证据。Counterfactual Effect-aware
候选仍等待Ordinary/Role-aware的实际结果后再决定，不据此结束研究。

本地全量Pytest结果为311 passed、1 failed/1861.72s；唯一失败是仓库外轻量导入子进程
超过120s，断言没有失败。该项原样单独复跑为1 passed/178.16s（含pytest启动），没有
放宽timeout或跳过断言。后续23项公平性/GPU争用/FM契约测试通过；Ruff check/format、
compileall、pip check及仓库卫生检查通过。实现提交 `fb23d2b` 已正常push至main；
[GitHub Actions 42374356968](https://github.com/Vincent-Wenhan/image-editing/actions/runs/42374356968)
的精简CPU与Torch CPU两条任务均成功，包含Pytest、Ruff、CLI和pip check。

历史真实生成图继续保留于 `outputs/qwen_cjt_q2_formal_gallery_20/paired_dev.png` 和
`failure_cases.png`，它们对应Q2的20-update对照，不能作为本次Q3的生成或收益证据。
待有空闲lease时，先在 `smoke_01` 完成2次更新、核验参数变化并用checkpoint-2真实重载
生成非零Train样本，再启动已准备好的两组120-step overfit及逐20步配对评估。

### 历史首次实现记录

- CPU role-loss/bridge targeted tests：已通过；真实全量回归和 CI 在代码提交前执行。
- 1-step smoke 的三次启动均未形成质量结果：第一次是 JSON manifest/CSV 混读，第二次是
  Dynamic Binary 绝对 control path 解析，第三次在共享 GPU 竞态中 OOM。失败回执分别保留在
  `outputs/qwen_cjt_q3_smoke_role/`、`outputs/qwen_cjt_q3_smoke_role_retry_02/` 和
  `outputs/qwen_cjt_q3_smoke_role_retry_03/`；没有修改第三方源码，也没有中断他人进程。

正式 overfit、full Train、Train/Dev 生成和比较结果待稳定空闲 GPU smoke 通过后登记。
