# 实验记录

时间以 Asia/Shanghai 为准。每条记录说明配置、输入、产物、实际结果与限制。

## 2026-10-21：Stage 1 初步对照完成

- Static/Dilated/Shuffled均完成150 steps/batch4，worker耗时1234.75/1230.83/1232.95s，峰值reserved均17.09 GiB；每组150次实际双图检查、160个有限张量，header step150/epoch2，无OOM或cleanup/telemetry错误。
- 三组固定12例/50 steps/CFG4完成，逐图生成合计450.31/456.73/450.22s，峰值reserved均16.39 GiB；完整阶段含初始化分别721.90/728.81/721.61s，不能直接把wall差异归因于算法FLOPs。
- 复用Stage 0 No-mask，完整配方、实际数据、采样预算与checkpoint SHA均再次通过；未重复生成或覆盖其结果。target MAE：No-mask0.055915、Static0.057758、Dilated0.056720、Shuffled0.074145。
- Static/Dilated没有改善No-mask的target MAE；Dilated外漂移略低且变化面积略少，Shuffled误差/漂移较大。单seed/12例像素诊断没有方法优越性结论；No-mask原cast顺序保留，也是后续应匹配的数值因素。
- 150-step Static在同checkpoint/源图/seed/50 steps下做Static/零图/Shuffle/重复：替换mask的输出像素MAE0.029828/0.026414，重复为0；常规推理与独立重载Static PNG相同，证明训练后仍消费条件。
- 补齐实际控制图画廊后，130项完整回归（111.77s）、Ruff、CLI/依赖检查通过。原数据、三组副本与固定版本外部checkout复核通过，GPU全部释放；详细SHA、失败、耗时、局限见Stage 1报告。

## 2026-10-21：实际控制图的可视化补齐

- 初次画廊沿用原始标注mask列，可能误导Dilated/Shuffled输入审查；新增实际控制图列，按记录尺寸重建，原No-mask布局不变。
- Static/Dilated/Shuffled的推理无target读取及可视化回归共7项通过；新视图另存各组 `gallery_controls/`，初次画廊保留。
- 目视抽查Static的dev_319096_t1镜面有白块残留，No-mask同例表现不同；这是探索性失败观察，不能当作编辑成功率或确定机制归因。

## 2026-10-21：同预算三组训练启动

- 执行代码 `46c5885`，三组150-step/batch4从同一Base独立开始：Static/GPU2、Dilated/GPU0、Shuffled/GPU1，UUID互异；容量门禁与完整预算校验已通过。
- 计划和完整配置/哈希在 `outputs/stage1_20261021/comparison/plan.json` 与各组 `budget.json`。固定12例重载/评估尚未完成。
- 新一轮本机128项测试（70.97s）、Ruff、依赖/仓库卫生、十个CLI和薄入口通过；远端CPU CI已恢复通过。

## 2026-10-21：Stage 1 编码与 Mask 消融通过

- 发现 33×33 PIL MaxFilter 使每次 Dilated 索引复核耗时较长，改用相同方形窗口的积分图；半径1/2/16及边界的15项条件回归通过。5张真实512² mask逐像素相同，CPU均值0.583s→0.00883s，仅为本次微基准，不代表GPU吞吐。
- 2-step/batch4容量检查完成：192.63s、峰值reserved17.09 GiB、两个双图batch检查通过、checkpoint有限；相对于空闲A6000保留充分余量，正式比较固定batch4。
- 三组正式配置/独立数据和条件索引已准备，预算验证均通过：200/30、150 steps、batch4、seed42，完整 train/network/model/sample 与 Stage 0 No-mask 一致，原 config/checkpoint 实际哈希相符；尚未启动正式训练。
- 同一 2-step adapter、固定 dev_140513_t1/seed3662387196/4 steps/CFG4，真实 encoder 接收两图；joint attention 的 T=0/10/20 各 1024 tokens。
- Static/零图/Shuffled 保持 source/noise 不变，mask token 和输出发生变化；Static repeat PNG 完全一致。worker 110.64s、wall126.63s，峰值 reserved16.56 GiB；仅证明条件接入，未证明正确空间编辑。
- FP32 算术 source/mask 相对 RMSE0.000306/0.000414，BF16为0.007261/0.021445；打包形状/位置一致，但数值非字节相等，未调整既定容差。
- 本机125项测试通过；远端 CPU CI 暴露新增fake backend测试漏mock版本查询，缺Torch metadata，已修复测试隔离，等待新checks。
- 正在进行独立2-step/batch4容量检查；三组正式150-step对照尚未开始。详细状态见 `docs/stage1_report.md`。

## 2026-10-21：Stage 1 联合条件实现与 CPU 回归

- 修复后 `joint_debug_02` 完成 2 steps/batch1，194.61s，峰值 reserved 15.04 GiB，两个实际双控制 batch 检查通过，160 个 LoRA 张量有限；GPU 已正常释放。编码与消融尚待执行。
- 数值对照采用相同 BF16 权重量化后的 FP32 算术，再检查真实 BF16；不会把未量化原始 VAE 与已经 BF16 加载的 pipeline 直接混比。
- 第一次实际 CLI 配置生成发现 metadata 中的 StrEnum 无法由 YAML safe_dump 序列化；内存配方测试没有覆盖写盘，已改为普通字符串并增加写出/重读回归。训练随后在缺配置预检退出，没有加载模型或执行 step；日志保留在 `outputs/stage1_20261021/smoke/train.log`，下一次使用独立配置/报告。
- 已实现 Static/Dilated/Shuffled 的纯二值变换和独立控制索引，SHA256、split 内 donor、图序与几何在配置、worker、推理、评估边界复核。
- 训练器调用边界强制每例双图、RGB [0,1]、mask 二值且与 target latent 几何一致；No-mask 保留原调用，flow loss/网络不变，第三方源码未改。
- 发现训练器先 cast BF16 再归一化，Diffusers 顺序相反；双参考推理限定作用域对齐训练顺序，结束/异常恢复，原 No-mask 处理器不变。
- 最终 120 项完整环境测试通过（44.02s），含实际 adapter 转发、target 读取拒绝、像素 cast、异常恢复与 snapshot 缓存隔离；Ruff/CLI/可复用 smoke 入口帮助通过。GPU smoke 与同预算对照尚未执行。

## 2026-10-21：PR 合并与 Stage 1 设计

- PR #1 在两组 GitHub checks 通过后正常 merge 到 `020c370`，保留原提交，无 force push。CI 为 95 passed/2 skipped，本机完整环境 97 passed。
- main 已 fast-forward；用户原 roadmap 移动与已提交内容 SHA256 完全一致，现由 main 跟踪，没有丢失修改。Stage 0 所有输出留在原架构 worktree。
- 从最新 main 建立 `feat/static-spatial-control`，独立 worktree/`.venv` 复用命名 Conda 依赖，原模型/数据/外部源码通过忽略的链接复用。
- 已阅读固定 CJT、ai-toolkit 双参考与 Diffusers 编码路径；设计见 `docs/stage1_design.md`。当前只完成设计，未实现空间控制或运行新 GPU 实验。

## 2026-10-21：架构分支合并门禁

- 远端 main `5c505e5` 是分支 `eaa30dc` 的祖先；无分叉冲突，diff whitespace 检查通过。
- Ruff check/format、97 项 Pytest（38.36s）、9 个 CLI 帮助及包/薄脚本入口、shell 语法、依赖和仓库卫生重新通过。
- 检查全部新增历史 blob（128 个），无权重、数据、图片、checkpoint、第三方或实验资产；最大 blob 21,223 bytes。
- 本地证据在 `outputs/merge_check_20261021/review.json`。完整 Stage 0 输出、checkpoint、耗时与报告继续留在原 worktree，没有重复运行或覆盖。
- 检查期间最初只有 SSH Git 身份，随后用户完成 GitHub CLI/API 认证。当前 PR 尚未合并，将核对远端 checks 后正常 merge；Stage 1 尚未开始实现。

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

## 2026-10-21 Stage 1 正常合并与后续设计

- PR #2 在 PR CPU CI success 后正常 merge，main `a03f168`；本地130 passed/62.75s、Ruff、CLI、pip check、卫生检查通过，历史78个新增blob无资产，未force push。
- 从最新main创建 `feat/role-data-and-evaluation` 独立worktree；设计先记录于 `docs/stage2_design.md`。
- 计划：constant双参考基线、显式No-mask预处理对照、30例固定验证、独立人工评分；随后最小Blender平移/阴影/背景及Oracle接口。尚未实现或声称研究收益；旧实验只读保留。

### 匹配双参考的无信息条件（已实现，GPU待运行）

- 实现constant_mask及显式No-mask training cast；旧No-mask默认调用与外层记录格式保留。
- 条件/CLI/编码等24项测试通过（41.94s）；包含constant的训练预算、配方、失败拒绝及旧推理记录30项通过（28.63s）。Ruff通过；没有把CPU检查记为GPU通过。

### 人工评估工具（已实现，真实评分待填写）

- 离线盲评导出/CSV导入实现成功度与无关内容保留0/1/2、失败类别及说明、reviewer；方法映射独立保存，实际图像/运行哈希复核，BOM兼容。
- 9项评估/CLI测试通过（23.56s），包括pending、匿名页面、评分往返、重复/缺项/越界/缺评审者/改图拒绝。测试评分仅为fixture，不是研究人工结果。
- constant的2-step/batch4真实训练已通过实时空闲检测和UUID租约启动，结果尚待完成。

### 显式预处理与参考数量的最小追踪（已实现，GPU待运行）

- reference_smoke追踪同一模型的legacy单参考、training cast单参考、constant双参考和重复采样；核对真实VAE/attention数量、T位置、source/noise摘要、全黑输入归一化、重复PNG及像素差异。仅验证协议，不作质量评价。

### Stage 2 数据实现与CPU验收（真实渲染待运行）

- 实现显式Cycles CPU场景、normal/no-shadow/occupancy/background线性证据、多个物理位移、兼容普通pair的manifest和独立T/D/P/valid标签。
- 从占据与shadow-ray反事实重算角色，阈值间隙/边界/无法解释变化ignore；验证背景显露、源图一致、数据哈希和场景组split。
- 18项CPU/CLI测试通过（18.23s）；fixture不当作真实Blender证据。正式软件安装与真实渲染尚待完成，首次软件入口403/慢链路续传日志保留。

### Oracle 数据与模型接口（已实现，GPU待运行）

- 独立监督与oracle loader，role RGB=one-hot T/D/P、ignore=黑；普通infer/edit继续拒绝，显式edit_oracle检查用途/哈希/几何/图序，不接收target。
- 32项场景/接口/旧推理回归通过（20.29s），另13项编码/后端/共享hook清理通过（15.14s）。删除target/反事实后可加载条件；仅CPU测试，真实角色GPU消费待运行。
- 共用生成流程和诊断hook，无loss/网络修改；未训练Role Predictor、Oracle adapter或Qwen。

### Blender本地运行库与版本探测

- wheelSHA验证、Cython/zstandard安装完成，首次bpy导入因缺libXrender失败；新输出保存安装日志。固定9个Ubuntu包版本/SHA本地提取，仅CPU渲染子进程追加库路径，父进程/CUDA任务不改动。
- 真实bpy4.5.3 LTS/build67807e1800cc导入与场景构建通过；版本严格按4.5.3数值tuple核验。20项相关测试通过（15.26s），真实多强度渲染待执行。
- Constant150步训练已complete：1153.01s，峰值reserved17.09GiB，固定30例重载正在运行。No-mask两种预处理及Static的30例推理完成，另两组Mask尚在运行。

### 首次真实场景渲染失败与边界修复

- render_01四场景×三强度完整渲染，worker8.22s。背景显露门禁拒绝train_cylinder_b/s001：两像素[55,47]、[64,47]误差0.013770 >0.01，均紧邻新轮廓。
- 原valid只排除union外边缘，漏掉重叠内轮廓。改为分别排除旧/新silhouette，误差阈值不变；补充重叠边界及failed/running数据拒绝回归，失败全部资产保留。
- 此前完整工程检查177 passed/71.82s，Ruff/pip/安装/嵌套CLI全部通过；本次边界修复另测后将用新目录真实重跑，不将失败写成通过。

### 独立数据副本的盲评兼容修复

- Constant为隔离缓存使用独立200/30副本；盲评现在分别验证各run自己的控制根和公共展示数据，而不改旧记录/降低哈希约束。另检查实际prediction尺寸/模式。
- 9项盲评回归通过（12.48s），包括constant副本与原数据比较、完整pending导入及预测尺寸伪报拒绝。
- render_02在分别排除新旧轮廓后真实通过：12对、四场景、6train/6dev，生成/验证12.11s，背景显露最大误差0.007452 <0.01，原render_01失败保留。

### 可选渲染依赖固定

- 将已实际安装的Cython3.3.0与zstandard0.25.0加入blender extra，bpy仍固定4.5.3；不改变模型依赖和原Conda环境。
- pip check通过，软件wheel与9个本地运行库实际SHA均已验证，CPU渲染与角色GPU接口已运行。

### 场景角色核验视图

- gallery新增显式show-roles，实际标签重建RGB与valid分列；未启用时旧布局兼容。31项场景/旧画廊/CLI相关测试通过（20.78s）；另两项无预测/含真实控制来源的组合视图回归通过（3.39s）。
- 本地role_gallery完成六个dev场景/强度的核验视图；目视确认身份全P、旧/新占据T、阴影迁移D、独立ignore边界。该目视核验不是人工模型质量盲评。
- 原Stage0/1四组共48张历史预测与新30例前12张逐字节相同；render_01/02的24张source/target PNG也逐字节相同，说明本次修复改变有效域而非渲染图像。原结果保留。

## 2026-10-21 本轮完整交付状态

- Stage1 PR #2正常merge main a03f168，分支从该最新main创建；无force push，Stage0/1 worktree和所有历史资产保留。
- Constant：2步/batch4 worker209.92s、reserved17.09GiB；150步/batch4 worker1153.01s（wall1161.16s），检查150个双参考batch、160张量有限，checkpoint SHA78ad1bd5494ae1a6e5c778835747a1c657fd687d9df0a804318ac64fab834463；无OOM或清理错误。
- 六组×30dev固定seed/50steps/CFG4完成。Constant/Static/Dilated Target MAE为0.056721/0.060950/0.057524；像素排序不等于成功度，目视摩托车替换样例中Constant未执行编辑而Static/Dilated替换为汽车。
- No-mask默认和training cast同checkpoint的30例输出平均MAE0.002380；原四组各12图共48图与新运行逐字节相同。并发controller head不能代替所有排队worker源码快照，相关限制写入补充报告，未虚称逐worker完全可追溯。
- 180项匿名真人评审包及空表导入完成，全部pending；失败候选选择/代理非盲观察独立记录，不计作真人评分或总体失败率。
- Blender render_01失败资产全留；修复内部轮廓valid后render_02完整通过：4组×3强度、6train/6dev，worker8.37s、总生成/验证12.11s，背景显露max0.007452<0.01；24张RGB与失败轮相同。
- Oracle真实GPU：128²/4steps，双VAE参考及T=0/10/20各64token；T/D置换改变条件/输出，source/noise不变、重复PNG一致。worker69.85s、reserved15.22GiB；仅无角色adapter的协议检查，不是质量上界。
- 最终184 passed/79.97s，Ruff check/format、pip check、editable、12CLI及嵌套/模块/薄脚本通过；wheel仓库外导入/自带配置/CLI通过。代码CI success(e5a7c51)，只读核对三个第三方checkout均固定且干净。
- GPU任务均经实时空闲检测与UUID租约，完成后8卡0MiB/0%，无计算进程。模型、数据、图像、checkpoint、环境、外部仓库仅在忽略目录。
- 当前未完成：真人评分、Oracle LoRA质量上界、多训练seed/更大scene-group/保留测试集；未实现Role Predictor、软Role模型或Qwen训练。新文字报告分别为stage1_review_report.md与stage2_report.md。

### 渲染采样说明补全（仅文档）

- 明确实际beauty/反事实/空场32 samples，emission-ID occupancy使用min(16,samples)=16；不能把两条路径抗锯齿覆盖视为精确相同。现有轮廓ignore与0.01门禁已覆盖该差异，实验与代码不变。
# 2026-10-22 角色研究验证设计

已读指定三份报告、架构及相关实现。PR #3 仍为 Draft、远端一致/CI success。
本轮先设计后实现，详见 `docs/role_validation_design.md`。现有180项真人评分为空，
不填充；新增3-reviewer导入/配对区间、分组公平性审计、三分割128→512场景，
再做Oracle/Static/Dilated/Dynamic Binary同数据双seed60步对照。当前均为计划。
启动GPU任务前重新检测并获取正常UUID租约。历史结果保持只读。

## 多评审工具实现

已实现panels/import/merge，按原字节保留评分与SHA，拒绝混合/重复reviewer，
至少3位完整独立声明前保持pending；paired按scene及reviewer bootstrap，分开两项
评分，NA显式缺失。12项人工评估回归通过（23.78s），Ruff通过；测试评分仅fixture。
真实180项仍空白，没有人工均值、区间或失败率。尚未执行本轮GPU实验。

## 场景变化与上界条件数据

已实现固定seed展开、scene级初始位置/光源/背景变化、可选test分割、完整snapshot，
以及独立Oracle/Dynamic Binary条件索引。58项相关回归通过（51.03s），128²真实
CPU渲染20组/60对通过，36/12/12分割，48.44s，背景误差max0.006304<0.01。
512²相同物理组正在验证；角色训练与GPU结果另记，当前不宣称上界质量。

## Stage 1 实际公平性与分组复核

Constant/Static/Dilated三组实际训练report/config/checkpoint/控制索引/manifest及
完整配方一致性审计通过。30例指令分为6类（只描述任务，不给成功分），实际图片
重算的诊断与旧metrics逐项一致。输出 `role_validation_20261022/stage1/audit.json`。
三份独立顺序的180项评审页面已导出；真实reviewer输入为0，合并结果pending3位。
28项协议/人工/worker回归通过（34.25s），未将fixture或代理观察作为真人评分。

## Oracle训练通路与512数据验收

已实现Oracle/Dynamic Binary训练/显式上界推理、实际one-hot batch校验，以及同预算
四组双seed编排和T/D/P分区诊断；保留原flowmatch/sigmoid、网络和loss。CPU测试证明
补丁前后loss/梯度一致，普通推理拒绝真值条件，标签消费不读target/反事实RGB。
第一轮全套193 passed/145.15s；新增协议28 passed/34.25s、分区指标2 passed。
512²真实CPU生成/重算60对，473.42s，背景误差max0.007431<0.01，未放宽阈值。
GPU smoke、容量、正式训练和重载尚未执行；下步冻结代码后启动，结果单独记录。

冻结前全量检查：199 passed/88.62s，Ruff check/format通过；新增上界runner、配方
漂移、角色指标与scene聚合均覆盖。512数据副本/控制图/配置正在CPU准备，不占GPU。

## CPU CI 环境修复

`6879c87`的GitHub最小CPU环境暴露测试隔离遗漏：上界runner的模拟backend测试未
模拟packages元数据，因CI不安装torch而失败（190 passed/7 skipped/2 failed）。
补齐fixture版本stub，生产执行代码不变；本机真实GPU smoke已在冻结worktree启动，
不会修改其源码或中断任务。outputs忽略规则同时覆盖独立worktree中的本地资产链接。

## Oracle 最小 GPU 训练完成

冻结执行6879c87，512²/batch1/2steps，worker143.7247s，峰值15.0219/15.0371GiB，
实际双参考校验2次，有限checkpoint SHA7fd99d95c42ffc6a069cb9e98a9bab4c02c59a66f80f9b36afbe4211df947ac6。
重载/容量/正式研究pending；具体状态见docs/role_validation_report.md。fe0029f双CI success。

## 编排边界检查

补齐preparing/prepared与完整method×seed矩阵门禁，避免提前启动不完整对照；
重载前再次核对训练配置摘要。新回归验证partial pilot允许、正式比较拒绝、重复身份
拒绝，Ruff通过。冻结GPU计算代码仍为6879c87；启动正式研究前单独复核全部10项。

## 容量、重载与评估证据绑定

Oracle重载smoke通过：512²真实两VAE输入，T0/10/20各1024tokens，T/D置换改变
control/output但source/noise不变，重复PNG精确一致；worker71.007s，reserved16.3945GiB。
batch4单卡2步通过：158.4285s，峰值15.0219/17.0859GiB，checkpoint SHA
24a92347ed1efc3b18b8c1c2fd294d0c861aee4429a69bebe6fc2fc8b004a390。
完整10项配置已复核，四方法双seed60步在四卡租约下运行，首轮GPU利用率100%、
约17GiB/卡，主机RAM仍有余量。固定batch4，不作运行中预算调整。
补充评估的run-LoRA/训练checkpoint摘要绑定与实际RGB/尺寸门禁；23项相关回归
通过（34.86s）。此前完整200 passed/92.45s，正式训练与质量结果尚未完成。

## 分区identity参考与最终CPU检查

新增T/D内“不编辑source”的误差参考，避免全图identity MAE掩盖编辑区失败；
保持原T/D/P指标、模型与训练预算不变。201项全套通过（111.49s），Ruff/CLI/依赖
及99项wheel文件的仓库外导入/配置检查通过。当前seed42四组各60步已完成并复核
checkpoint实际SHA/160个有限张量，seed43运行中；质量与真人结果仍pending。

## 场景采样范围配置收尾

将物体/光源/背景采样范围独立为YAML，CLI支持--variations，消除Python默认范围。
默认新展开20个scene物理参数与已冻结v1逐项完全相同，不重新渲染；增加固定范围、
无效边界、split字典顺序回归。正式四方法双seed训练均已完成，全部预算验证通过，
固定dev/test重载运行中。源码冻结6879c87与当前新采样器版本分别记录，未修改旧数据。
