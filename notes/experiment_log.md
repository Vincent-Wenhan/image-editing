# 实验记录

时间以 Asia/Shanghai 为准。每条记录说明配置、输入、产物、实际结果与限制。

## 2026-10-21：架构稳定性收尾开始

- 保持 `refactor/modular-architecture` 的目录设计，在原独立 worktree 修改。
- 启动时无活动项目 GPU 任务，8 张 A6000 均为 0 MiB /0%；不会中断其他任务。
- 发现设备补丁可重复叠加、失败清理/统计可遮蔽异常、评估未核实数据内容，以及固定 22 GiB 扩 batch 阈值。
- 设计与验收步骤见 `docs/architecture.md`；先补测试，再执行 20/150-step 与固定样本重载。旧权重和实验保留。

### 输入契约与评估校验

- 已增加 RGB、尺寸、有限数值、二值有效域及硬/软 T/D/P 标签约束，只验证接口，不实现新方法。
- 评估复核 manifest 及全部 source/target/mask 字节 SHA256；新增配置摘要与可选 Base/LoRA 预算比较，旧记录保持可读。
- 数据漂移、篡改配置、种子/尺寸/选择不一致和软场无效值等 CPU 回归已通过；完整训练尚未执行。

### 训练生命周期与容量规划

- 设备补丁幂等并绑定固定源码版本；原调用透传、设备变化、CPU loss/梯度/参数更新一致性通过。
- 失败清理、同步/cleanup 次级异常、报告写入失败不遮蔽原异常的测试通过；launcher 等待 worker 清理退出。
- 用实际租约卡容量、debug 峰值、最大 2 倍增长及 20%/4 GiB 余量替代固定 22 GiB 阈值；记录有效 YAML 与内存决策。
- 放大后 OOM 只允许一次新实验名降级，其他失败不重试。该阶段 97 项 CPU 回归通过，完整训练尚未执行；随后结果如下。

### 完整 Stage 0 验收完成

- 执行代码为 `26bc4c2`，运行目录是 worktree `outputs/stage0_stability_20261021/`；开始 03:50:24、结束 04:14:45（Asia/Shanghai），控制器报告 1460.10s。
- 只复制现有 200 train /30 dev 数据以隔离缓存，manifest 字节 SHA256 不变；训练前后原数据与副本均通过配对/哈希验证，跨 split session/source 交集为 0。模型没有移动或重下。
- 旧 12 例 Base 校验模型 revision、无 LoRA 状态、数据哈希和默认采样配置后复用；新旧 Base metrics.json 一致。
- 20-step debug：batch 2 /512² /rank 16 /seed 42，199.58s，峰值 allocated 15.02 /reserved 15.04 GiB；160 个 LoRA 张量有限，header step=20，checkpoint SHA256 `138ba8e4ca9ec9bef29199ca55146f883396df5f6c1d0e7d5c392424d43ac2b1`。
- 150-step Edit LoRA：独立从 Base 开始，batch 4，其余配方不变；787.24s，峰值 allocated 15.02 /reserved 15.10 GiB；160 张量有限，header step=150 /epoch=2，SHA256 `911bb69a934768805c942a635f50c3d13c6bea1717706d82cdd9e9b745fe4b31`。
- batch 决策基于实际租约卡可用 47.99 GiB、debug 实测 15.04 GiB，保守估计放大后 30.07 GiB，预留 9.60 GiB；有效配置与决策文件保留。没有实际 OOM、降级或清理/统计错误。
- 同样 12 例、50 steps /CFG 4 /512² /稳定 sample seed 的重载完成；加载 SHA256 与最终 checkpoint 一致，12 张 PNG 均与 Base 不同。逐图生成合计 317.94s（均值 26.50s），含加载全阶段 390.92s，峰值 reserved 16.26 GiB。
- 配对评估与画廊完成。Target MAE：0.073098 →0.055915；mask 外漂移 proxy：0.054625 →0.034876；变化面积：0.302519 →0.188654。误差较低但变化也较少，identity target MAE 为 0.030950，不推断编辑成功率提高。
- 结束后实时确认所有 GPU 为 0 MiB /0%、无计算进程，租约已释放。旧失败记录保留；OOM/cleanup 异常恢复只经 CPU 注入回归，未人为制造 GPU OOM。
- 详细证据为本轮 `verification.json`、`status.json`、训练报告、`lora_inference/metrics.json` 和两组画廊。所有资产和日志留在本地；未开发 Static Mask、Role Predictor 或 Qwen。

## 2026-10-21：架构重构开始

- 在独立 `refactor/modular-architecture` worktree 开始；保留主工作区和命名 Conda 环境。
- 完整阅读研究文档、代码与测试；原有 19 项 CPU 测试通过。启动前没有活动项目任务，8 张 GPU 空闲。
- 发现旧验收已完成 200/30 数据准备和 12 例 Base 推理；debug 在第一步出现 CPU/CUDA 条件张量设备错误，LoRA 尚未验收。
- 设计与分步验收见 `docs/architecture.md`；本条不是重构或训练完成声明。

## 2026-10-21：模块拆分与推理行为核验

- 删除旧包层，`src` 作为明确 package；加入输入/监督/控制/后端契约，业务拆入八个职责模块。
- 训练和验收脚本变为薄入口；ai-toolkit 使用安装后的模块 worker，不再修改自有 `sys.path`。
- 独立 worktree 的 `.venv` 复用 `image-edit` 已安装依赖，独立 editable 安装；原 Conda 安装不变。
- 原 19 项测试迁移并补充 CLI 默认值快照和数据/配置/协议/记录测试后，42 项测试通过；Ruff 与依赖检查通过，最终验收后续更新。
- GPU 租约下完成 1 例 Base 推理：512²、50 steps、CFG 4，23.36s，峰值 reserved 15.95 GiB。
  相同样本与 seed 的 PNG SHA256 与旧验收完全一致；证据在 worktree `outputs/refactor_validation/base/`。
- 旧训练的 VAE CPU/CUDA 设备问题独立处理，当前还没有完成重构后的训练验收。

## 2026-10-21：固定训练器的设备适配

- 原因：上游准备训练时把 VAE 移到 CPU，FLUX 随后编码原始 reference，拼接时与 CUDA 主 latent 不一致。
- 适配只在固定 Klein 类的公开方法边界恢复 VAE device；原 dtype、forward、flow loss 和所有训练参数不变，第三方源码未修改。
- 新的 8 train /2 dev 文件副本用于隔离缓存，原数据与实验保留。两步 batch 2 /rank 16 /512² 训练完成，160 个 LoRA 张量均有限。
- 总耗时 72.25s（含模型加载/缓存），峰值 reserved 15.04 GiB；证据为 worktree `outputs/refactor_validation/train_report.json`。
- 设备适配的 6 项 CPU stub 测试通过；同样本 50-step LoRA 重载完成，27.08s，峰值 reserved 16.26 GiB。
  `run.json` 的 LoRA SHA256 与保存的 checkpoint 一致；20/150-step 完整验收不计为通过。

## 2026-10-21：架构重构最终验收

- 完整环境 50 项测试通过；独立无 torch CPU 环境 49 项通过、1 项 checkpoint 张量测试按依赖跳过；完整环境执行了该测试。
- Ruff check/format、依赖检查、九个子命令帮助、薄脚本入口、shell 语法、仓库卫生均通过；editable 和 wheel 安装后从仓库外导入/加载配置/CLI 通过。
- 只读校验原 200 train /30 dev 数据，跨 split session/source SHA256 交集为 0。复制旧 12 例 Base 产物再评估，metrics.json 与旧报告完全一致；生成了新画廊。
- Base 一例重跑与旧 PNG 逐字节一致；两步 LoRA 训练、160 个有限张量和一例同配置重载均通过。所有 GPU 操作先检查空闲并获取 UUID 租约。
- 固定第三方仓库均无源码修改，原模型/数据/实验保留。一次性 CPU 环境、构建/wheel 文件清理；原始验证日志与配置保留在 worktree `outputs/refactor_validation/`。
- 本次交付是工程重构及设备兼容修复；未运行完整 20/150-step 训练或新的空间控制。下一步先完成完整 Stage 0 基线验收，再进入 Static Mask。

## 2026-10-20：Stage 0 准备（进行中）

- 初始仓库为空。保留用户原始 roadmap；审阅 starter kit，建立维护中的包和开发规则。
- 硬件：8× RTX A6000，单卡 49,140 MiB；驱动 535.230.02；初始检查全部空闲。约 38 TB 可用磁盘、149 GiB RAM。`nvidia-smi topo -m` 显示 SYS 互联，并有 hwloc NUMA 拓扑警告，因此先单卡，不假定有 NVLink。
- 环境：创建项目独立 Python 3.11.17。下载 PyTorch 2.7.1 / CUDA 12.6 依赖。CUDA 13 主分支配方不适用；固定 ai-toolkit `a055947`。其原始 Diffusers pin 早于 Klein pipeline，推理/训练统一改为 Diffusers 0.37.0（`8328e50`），兼容性待真实执行确认。
- 模型：按 `configs/assets.lock.json` 下载 Base、训练 VAE 和 Qwen3-4B 文本编码器。未下载 Qwen-Image-Edit、Inter-Edit 全量集或 MagicBrush test。
- 数据：目标为 200 train / 30 dev first-turn；PyArrow 26 导致 NumPy 1.26 ABI 兼容错误，明确固定 PyArrow 19.0.1 后重试。
- 检查：11 项 CPU 单元测试通过；Ruff 检查通过；GPU 启动器实际检查空闲 GPU 0 并为无 CUDA 计算的子进程设置其 UUID。
- 原始日志：`outputs/logs/`。下载完成不等于推理完成；当前尚未记录真实 CUDA 推理或 LoRA 训练成功。
- 代码已按 Conventional Commits 提交并使用现有 SSH 身份推送；HTTPS 无凭据，不在仓库保存任何凭据。

本条在环境、数据、推理和训练验证结束后更新；指标结果以对应 `run.json` / report 文件为证据。

## 2026-10-21：命名环境、真实数据核验与传输恢复

- 用户要求使用新的命名 Conda 环境：已创建 `image-edit`，实际路径 `/gluster/envs/users/jiangwenhan/conda-envs/image-edit`，Python 3.11.17。初始项目前缀环境用于引导；完成迁移后清理重复环境。
- 当前 CUDA 验证组合改为 PyTorch 2.6.0+cu118；已在启动器选出的空闲 RTX A6000 上实测 BF16 matmul，结果全部 finite。原计划 2.7.1 的下载多次超时，没有将其写成验证通过的版本。
- Diffusers 0.37.0 commit 为 `d791c5c024014784df4a8dac2601e19fb4d300fc`；Klein pipeline 和 torchao 0.10.0 的配置类已成功导入。训练环境的完整依赖检查仍在进行。
- 第一份 train parquet（173 个编辑步骤）完成 SHA256 校验。其 89 个 first-turn 样本已导出到本地 `outputs/data_audit/train/` 并生成画廊，前 30 组 source/target/mask 已逐页检查，几何与反向 alpha 语义一致。这是抽查集，不是最终 200 train /30 dev 验收结果。
- RGBA 原始 mask 的透明区才是编辑区域，source 500² 与 target/mask 1024² 需要统一尺度。已修复并通过专门回归测试。部分 target 有原始生成水印，部分指令/目标质量有限；像素误差不能替代编辑成功评价，后续正式研究应预先定义质量筛选规则。
- 训练 VAE 已下载；Base 权重仍在传输，ModelScope 作为传输镜像，逐文件匹配 Hugging Face 固定 revision 的 SHA256。此前整文件和 HF Transfer 传输出现超时及解码失败，现采用可恢复的 4 MiB 分块并校验最终完整文件。
- 18 项 CPU 测试通过，覆盖 Mask 编码、配对/泄漏、GPU 锁、分块续传与坏校验和拒绝。所有权重、图片和下载中间块均在忽略目录中。
- 当前尚未宣称真实模型编辑推理或 LoRA 训练完成；以随后产生的 `run.json` 和训练 report 更新验收状态。

## 2026-10-21：环境验收完成，后续步骤已排队

- `image-edit` 的 `pip check` 通过；在该命名环境内完成 Klein 导入/BF16 GPU 检查和 ai-toolkit `ExtensionJob` 配置构建。为上游 CLIP 导入固定 setuptools 69.5.1；为上游 eager 模型注册表补齐 PyAV 16.0.0。Conda 和 pip 锁文件已保存。
- 初始可用集为 `data/magicbrush_smoke/`（89 train /30 dev），来源为固定 revision 和官方 split，session/source hash 交集均为 0。最终目标 200 train /30 dev 继续下载，不将二者混写。
- 训练 VAE 完整文件已校验。Base 仍在下载；稳定解析 URL、分块缓存和断连后的字节续传已经实现，保留已收到的数据以减少重传。
- 已通过 19 项 CPU 测试及 Ruff 检查；LoRA wrapper 会验证保存的 adapter 张量存在且全部 finite，再记录 checkpoint SHA256。
- 下载与验收进程见 `outputs/background_jobs.json`；`outputs/stage0/status.json` 记录实际状态。后续自动执行 12 例 Base 推理、20-step debug、150-step LoRA、同配置重载和诊断。资产不完整时保持等待，不占用 GPU。
- 已清理临时脚本、下载引导 wheels、废弃探测文件和重复的项目前缀环境；只保留命名 Conda 环境。原始用户 kit 压缩包保留在本地且不提交。
- 本记录描述启动时的真实状态；完整模型推理/训练是否通过，以验收状态和各阶段 report 为准，目前没有研究性能提升结论。
