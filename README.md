# Image Editing Research

PR #5已正常合并main `676242b`。新分支feat/role-feature-diagnostics先诊断独立特征
为何没有稳定收益，复用原checkpoint/Dev，新增训练更新固定为0；不扩大训练。
残差/BF16/时间步/通道干预与一个最小修复的预注册见
[机制诊断设计](docs/role_feature_diagnostics_design.md)。下文保留上一轮实际结果。

机制诊断入口（不训练，GPU子进程由编排器申请UUID租约）：

```bash
python -m src.inference.feature_diagnostic_study prepare --existing-study <old-study> --output <new-study>
python -m src.evaluation.feature_encoder_diagnostics --study <new-study> --output <new-cpu.json>
python -m src.inference.feature_diagnostic_study run --study <new-study> --phase smoke
python -m src.inference.feature_diagnostic_study run --study <new-study> --phase diagnose
```

资源预检或执行失败后使用`--attempt 2`等新编号；旧尝试保留，已完成的seed从校验过的
receipt复用，完整phase禁止覆盖。固定状态回放还校验与原轨迹预测逐元素一致。

语义代理评估与非盲叠图（不生成真人评分）：

```bash
python -m src.evaluation.feature_diagnostic_analysis --study <new-study> --phase existing \
  --original-analysis <verified-original-summary.json> --output <new-semantic-analysis>
python -m src.evaluation.feature_diagnostic_gallery --summary <new-semantic-analysis>/summary.json \
  --output <new-gallery>
```

模板位置先用source/target/background校准，弱分离或无可信匹配会弃权。重复候选、
D阴影创建/移除与暗区IoU、P边缘漂移是合成场景代理，不是人工成功率。

Phase A真实4B smoke、双seed固定状态诊断、最小修复smoke与96图对照已完成。
弱角色差分及BF16舍入有证据，但居中/能量匹配没有稳定T/D收益，当前修复No-Go，
不扩大训练。两个seed三人盲评空包已就绪，真实评分0条/pending；Test继续封存。
选择依据、唯一算子及不变门槛见[修复预注册](docs/role_feature_repair_design.md)，
实际状态见[诊断报告](docs/role_feature_diagnostics_report.md)。

冻结残差改进（必须先完成Phase A；不训练，不改变默认推理）：

```bash
python -m src.inference.feature_repair_study prepare --study <new-study>
python -m src.inference.feature_repair_study run --study <new-study> --phase repair_smoke
python -m src.inference.feature_repair_study run --study <new-study> --phase compare
python -m src.evaluation.feature_diagnostic_analysis --study <new-study> --phase compare --output <new-analysis>
python -m src.evaluation.feature_repair_analysis --study <new-study> --summary <new-analysis>/summary.json \
  --output <new-analysis>/gates.json
```

四臂共享每例两次冻结encoder预计算；记录真实GPU残差文件SHA，原缓存必须精确
复现旧PNG。新run保持原协议并显式声明residual_intervention，人工工具校验实际
字段、encoder、公式与缓存后才能导出；不会自动填写评分。
252项本机CPU回归（138.39s）、Ruff、55 CLI通过；修复真实4B smoke110.64s通过，
96图比较两seed604.64/606.35s，allocated15.598/reserved16.266GiB，实际SHA与门槛复核。

上一轮分支 `feat/role-feature-conditioning` 从正常合并PR #4的main `b73492f`创建。
独立于Image VAE的角色特征最小试验设计见 [预注册](docs/role_feature_design.md)。
2026-10-23已完成真实4B前反向/有限梯度/条件置换/零条件/成对重载验收、batch4
容量及四方法×双seed×20updates训练。232项本机测试、Ruff、43项CLI和依赖检查
通过；远端CPU CI通过（没有Torch的检查明确跳过）。96张固定Dev预测与实际SHA
评估完成，两个seed均未通过预设门槛：**当前注入/20-update预算No-Go扩大训练**。
完整指标、区间与案例见 [验收报告](docs/role_feature_report.md)，GPU消费成功不代表方法收益。
旧负面结果、全部资产与真人pending状态保留，Test封存，不扩大训练。
[PR #5](https://github.com/Vincent-Wenhan/image-editing/pull/5)记录本轮实现与真实验收。

首轮GPU暴露固定上游TrainConfig不保留seed的兼容问题，已从已校验YAML显式传入，
同seed初始化/训练轨迹一致且两seed独立；原0-batch失败记录保留。执行源码冻结b256692，
新资产在outputs/role_feature_20261023/study_01，未改loss或第三方源码。
身份门槛使用真实s=0预测target_mae，避免与数据自身source-target距离混淆。
角色方案仅为显式Oracle上界；三人盲评按seed分别准备48任务/人，0条真实评分。

以下为已完成的 **Role Conditioning Failure Analysis**，从已合并 PR #3 的 main
`a647522` 开始。[预注册设计](docs/role_failure_analysis_design.md)规定新 Dev 和封存
Test、旧 checkpoint 干预、真人评审准备及最小 Geometry 上界；尚未运行的新实验
不能引用前轮结果充当验证。

本轮真实运行完成：新512数据20组、旧checkpoint的120次诊断生成、双seed各20-step
Geometry/Dynamic Binary训练及48张新Dev预测。彩色复制/重复物体再次出现；灰度V2
虽改善D/P，T门槛在两个seed均未通过，当前编码No-Go，不扩大训练。四个新身份
场景未目视复现旧大蓝块，但分布外P换色干预出现红边栏。新Test封存，两个三人
盲评包均0条真实评分/pending。
完整时间/显存/SHA/区间和风险见 [实际报告](docs/role_failure_analysis_report.md)。

最小V2接口已实现 `oracle_geometry`：同双参考，P/ignore黑，旧独占64、新独占192、
重叠128、依赖255的三通道灰度；仅显式oracle，不是预测器。原loss/网络不变，
独立特征的当前实现与验收见下节。真人使用步骤见 [评审指南](docs/human_review_guide.md)。

独立特征入口（显式Oracle、固定尺寸、仅Dev）：

```bash
python -m src.training.feature_study prepare --data-root <verified-scenes> \
  --cohort <sealed-cohort.json> --output <new-study>
python -m src.training.feature_study run smoke --study <new-study>
image-edit gpu -- python -m src.inference.feature_smoke --data-root <verified-scenes> \
  --cohort <sealed-cohort.json> --lora <smoke-lora> --encoder <paired-features> \
  --output <new-study>/gpu_smoke
# GPU门禁成功后才允许 capacity → train → infer，不启用自动batch扩大。
python -m src.training.feature_study run capacity --study <new-study>
python -m src.training.feature_study run train --study <new-study>
python -m src.training.feature_study run infer --study <new-study>
python -m src.evaluation.feature_analysis --study <new-study> --output <new-summary>
```

RoleFeatures为5×H×W浮点单位图；16像素面积池化、类别/坐标编码、零初始化残差。
编码器独立于Image VAE，成对权重绑定LoRA SHA，缺失或错配明确报错。
当前只支持新目录单卡独立训练与最终重载，不支持中断续训或多卡DDP。

```bash
# 先生成并封存新的三分割数据，再准备最小匹配实验。
python -m src.training.role_study prepare --data <new-scenes> --output <new-study> \
  --protocol configs/experiments/role_geometry.yaml --cohort <cohort.json>
python -m src.training.role_study run --output <new-study> --phase smoke
image-edit gpu -- python -m src.inference.oracle_smoke --control oracle_geometry \
  --data-root <new-scenes> --lora <smoke-checkpoint> --output <new-study>/oracle_reload_smoke
# 完成重载后依次 capacity、train、infer；infer严格只用新Dev，不运行封存Test。
```

`src.datasets.holdout.seal_cohort` 检查与历史场景/源图SHA无重叠并写不可覆盖摘要；
失败诊断 `python -m src.inference.role_diagnostics --help`，必须通过GPU租约。
`python -m src.evaluation.role_failures --help` 分开记录配色敏感性、旧区残留距离与
新区target误差，这些是像素诊断，不能代替物体计数和人工编辑成功率。

已完成研究轮次正式名称：**Causal Role Data and Oracle Feasibility Study**。
PR #3 的源码、资产、统计与结论复核见 [合并审计](docs/causal_role_feasibility_audit.md)。
彩色 Oracle Role 的负面结果及全部历史资产保留；后续失败分析独立开展。

2026-10-22：已完成三评审者导入/配对统计工具、Stage 1公平性分组审计、20个独立
train/dev/test场景组的128→512渲染，以及四方法×双seed的60-step LoRA与192张
固定评估。202项测试、Ruff、CLI、wheel检查通过；真实评分仍pending。
**当前彩色Oracle Role编码No-Go**：两seed的T误差明显高于Dynamic Binary，且有
控制图外观复制/重复物体/身份破坏。暂缓Role Predictor。完整数字、区间、checkpoint
和限制见 [实际报告](docs/role_validation_report.md)，预定预算见
[研究设计](docs/role_validation_design.md)。

Stage 1 已通过 [PR #2](https://github.com/Vincent-Wenhan/image-editing/pull/2) 正常合并。
2026-10-21的历史里程碑：本分支完成双参考无信息基线、六组各30例固定
验证、180项离线人工评审包，以及最小Blender反事实数据和显式Oracle Role接口。
Constant 150步训练、12对真实CPU渲染/标签校验和Oracle GPU协议检查已通过；
当时184项测试、Ruff、CLI、editable/wheel检查通过，尚未训练角色adapter。详见
[补充评估报告](docs/stage1_review_report.md)、[Stage 2实际报告](docs/stage2_report.md)
和 [设计](docs/stage2_design.md)。所有历史结果保留，新资产仅存本地outputs。

Stage 0 架构已通过 [PR #1](https://github.com/Vincent-Wenhan/image-editing/pull/1)
合并。Stage 1 的联合条件、130 项回归、GPU smoke、三组150-step训练及固定12例
评估已完成。初步像素诊断未显示优于No-mask。见 [设计](docs/stage1_design.md) 和
[实际报告](docs/stage1_report.md)。

Mask 画廊新增 `annotation_mask` 与 `control_input` 两列，分别展示评估用原标注和模型
实际收到的控制图；No-mask 的原四列保持不变。初次画廊保留，新视图另存 `gallery_controls/`。

以 **FLUX.2 Klein Base 4B** 建立可复现编辑与 LoRA 基线，随后研究随编辑强度变化的空间角色预测。方法改进尚未验证。

- [方案评审](docs/proposal_review.md)：可行性、贡献边界、标签定义与必要对照。
- [Stage 0 设计](docs/stage0_design.md)：当前准备工作的范围和验收。
- [Stage 0验收记录](docs/stage0_report.md)：已完成的训练、重载与历史失败。
- [原始 roadmap](docs/Image_Editing_Preparation_and_InterEdit_Roadmap.md)：研究输入与长期路线。
- [架构重构设计与验收](docs/architecture.md)：模块边界、兼容约定与实施步骤。
- 稳定性收尾与 20/150-step、固定 12 例重载验收已完成；真实结果见 Stage 0 报告。
- [Inter-Edit 代码走读](notes/inter_edit_code_review.md)：CJT 张量流及 FLUX 适配边界。
- [开发规则](AGENTS.md)：Conventional Commits、500 行限制、文档同步和 GPU 规范。

## 环境与资产

从项目根目录执行，Python 3.11。服务器驱动为 535，使用已通过 BF16 CUDA 运算检查的 PyTorch 2.6.0 / CUDA 11.8，实际实验状态以实验记录为准。

```bash
bash scripts/setup_environment.sh
conda activate image-edit
pip install -e '.[dev]'
# 仅当官方站点不可达时使用镜像；不写入系统配置。
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
image-edit download --asset all
image-edit prepare --train 200 --dev 30
image-edit validate
image-edit gallery
image-edit gpu -- image-edit env --cuda
```

下载固定 revision 见 [assets.lock.json](configs/assets.lock.json)。`base` 同时包含 Diffusers 与 BFL 原格式，后者供 ai-toolkit 使用；`training_text` 是 Klein 的 Qwen3 文本编码器，`training_vae` 是训练器需要的 VAE，不是第二阶段 Qwen-Image-Edit 模型。

`training_text` 通过本地链接复用 Base 附带的 Qwen3 权重和 tokenizer，不重复下载另一份文本模型。网络不稳定时可使用经过 SHA256 校验的分块下载：

```bash
image-edit download --asset all --transport ranges --payload-mirror modelscope
image-edit prepare --transport shards
```

ModelScope 只作为 Base 权重的传输镜像；每个权重必须匹配 Hugging Face 固定 revision 的 SHA256。数据只下载满足样本数所需的前几个 parquet 分片，并保留在本地 `data/raw/`。

MagicBrush 使用官方 train/dev、每个会话的第一轮，固定上游顺序；不接触 test。实际 source 为 500×500，target/mask 为 1024×1024；先检查宽高比，再统一长边 512。RGBA mask 用反向 alpha 提取编辑区，普通灰度 mask 才使用白色区域。导出保存原始与输出尺寸、mask 编码、样本身份、指令及 SHA256，严格检查配对和跨 split 泄漏。重复执行仅复用通过校验且 revision/数量/预处理一致的完整 split。

## 基线推理与诊断

```bash
image-edit gpu -- image-edit infer --limit 12 --output outputs/base_inference
image-edit evaluate --predictions outputs/base_inference
image-edit gallery --predictions outputs/base_inference --output outputs/base_gallery
```

默认 Base 50 steps、CFG 4、长边 512、BF16、模型常驻 GPU；每个 sample ID 有稳定 seed。新实验用新输出目录，保留 `run.json`、逐图结果与诊断，避免静默覆盖。画廊为本地 HTML 和 contact sheets。MAE、变化面积、mask 外漂移只是诊断指标，不是编辑成功率或物理因果准确率。

评估会复核 manifest 与 source/target/mask 的真实 SHA256、尺寸、样本顺序和种子。
新记录带配置摘要；配对比较可加 `--reference-run outputs/base_inference` 强制采样预算一致。
输入、监督、硬标签与软角色概率场的约束见架构文档；Stage 0 默认仍无空间条件。

## Stage 1 联合 Mask 条件

本机独立开发目录为 `/home/jiangwenhan/project/image-editing-static-spatial-control`。
进入该目录后先激活 `image-edit` Conda，再激活 `.venv`，使用此分支的editable安装；
原 main 和 Stage 0 worktree 各自保留。新环境仍可直接 `pip install -e .`。

复用 No-mask 时，`src.evaluation.budget.verify_training_budget` 校验原报告、配置、
checkpoint 和真实数据，比较完整训练参数与更新预算；结果保存在每组 `budget.json`。

`prepare-controls` 支持 `static_mask`、`dilated_mask`、`shuffled_mask`，生成独立的
RGB 二值控制图和 SHA256 索引。Shuffle 固定完整 split 的 donor，不跨 train/dev；
Dilated 默认 16 px 方形核半径。研究运行先复制验证过的数据以隔离训练缓存。
方形膨胀采用与 PIL MaxFilter 等价的积分图实现，避免大核校验拖慢 GPU 任务预检。

```bash
image-edit prepare-controls --kind static_mask --output outputs/controls/static
image-edit train-config --control static_mask --control-root outputs/controls/static \
  --name static_debug --steps 2 --output outputs/configs/static_debug.yaml
image-edit gpu -- python -m src.training outputs/configs/static_debug.yaml \
  --report outputs/static_debug_report.json
image-edit gpu -- image-edit infer --control static_mask --control-root outputs/controls/static \
  --lora outputs/lora/static_debug/static_debug.safetensors --output outputs/static_inference
```

训练与推理均按 `[source, mask]` 逐图 VAE 编码和联合 attention；不混合最终 RGB，
不将 target 当参考图，不使用 mask 加权 loss。缺图、错序、非二值或几何不匹配会失败。
实际 GPU 验收与对照以 Stage 1 报告为准；条件敏感性不等于编辑质量提升。

三组训练/12例重载/配对评估均已完成，每组最终checkpoint为150 steps、160个有限张量。
Static/Dilated/Shuffled的target MAE分别0.057758/0.056720/0.074145，复用的No-mask为
0.055915。预算、耗时、显存、失败、数值差异和下一步限制均在报告中，不据此宣称方法有效。

## Edit LoRA

```bash
image-edit train-config --steps 20 --name flux2_debug \
  --output outputs/configs/debug.yaml
image-edit gpu -- python scripts/train.py outputs/configs/debug.yaml \
  --report outputs/lora_debug_report.json
# 测量显存/吞吐后调整 batch，使用新实验名。
image-edit train-config --steps 150 --batch-size 1 --name flux2_magicbrush_smoke \
  --output outputs/configs/smoke.yaml
image-edit gpu -- python scripts/train.py outputs/configs/smoke.yaml \
  --report outputs/lora_smoke_report.json
# 将实际保存的 checkpoint 路径传给 --lora，复用相同样本/seed/步数。
image-edit gpu -- image-edit infer --lora outputs/lora/flux2_magicbrush_smoke/flux2_magicbrush_smoke.safetensors \
  --output outputs/lora_inference
```

训练 wrapper 校验上游 commit，将上游公开 class attribute 指向本地 Qwen3 权重，并强制离线加载；不修改外部仓库。短训练只验收 forward/backward/checkpoint/reload，不能预设质量提升。

`python scripts/train.py ...` 保持兼容，也可使用 `python -m src.training ...`。
训练器在自己的目录中通过独立 worker 导入，父进程的导入路径与工作目录保持稳定。
固定训练器在缓存后把 VAE 留在 CPU；本地适配在 reference 编码前恢复其设备，
保留原 dtype、forward 和 loss。现已通过 200 个训练样本上的 20-step debug、
150-step 训练、checkpoint 有限性检查与固定 12 例 LoRA 重载评估。

也可运行 `python scripts/run_stage0.py --output outputs/stage0_new_run`，等待固定资产准备完成后自动执行上述工程验收；状态写入该目录的 `status.json`。每次 GPU 启动仍会检查空闲卡，下载等待阶段不占用 GPU。新验收使用新的输出目录。

可加 `--base-run <已完成的Base目录>` 复用经版本、哈希与采样配置检查的固定样本。
20-step debug 后，下一次训练在实际租约卡上保守规划 batch，保留显存余量；
实际配置、决策和失败均单独记录，放大后 OOM 最多以新实验名降级一次。
设备补丁幂等；worker 失败时执行清理并保留原始异常、统计及报告。

## 开发与目录

```bash
pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
image-edit --help
python scripts/check_repo.py --all
git add <本次改动>
git commit -m "feat(scope): concise description"
git push
```

`src/` 本身是 Python package，使用 `src.xxx` 绝对导入；模块入口为 `python -m src`。
`src/core/` 管理数据契约、配置与记录；`datasets/` 管理数据，`conditioning/` 管理控制图契约；
`backends/` 封装 FLUX.2，`inference/` 与 `training/` 组织执行，`evaluation/` 评估与可视化，
`infra/` 管理下载、GPU 和环境。`scripts/` 保留轻量兼容入口。
完整目录、依赖、数据流、旧新映射和扩展指南见 [architecture.md](docs/architecture.md)。

配置分为 `configs/assets.lock.json`（模型/数据 revision）、`configs/datasets/`（数据选择）、
`configs/experiments/`（验收预算）和原有 `configs/flux2_edit_lora.yaml`（完整训练配方）。
CLI 从这些配置读取默认值，原有命令行覆盖有效。采样默认值复用配方的 `sample`，不在 Python/shell 中另写一份。
从仓库根执行默认命令；在其他目录使用已安装的 CLI 时，显式传入数据、模型和模板路径。

`EditInput` 与 `EditSupervision` 独立，推理只读取 source/instruction；`ControlBundle` 标记控制来源，
普通推理拒绝 oracle。FLUX 已接入 Static/Dilated/Shuffled Mask；Soft/Role/Qwen 仍未实现。

Stage 0 稳定性收尾的 97 项 CPU 回归、Ruff 与 CLI 检查通过；20-step /batch 2 和 150-step /
batch 4 均保存了有限 checkpoint，固定 12 例重载和配对校验完成。此前 editable/wheel
安装及 Base 单例字节一致验证保留。新证据在 `outputs/stage0_stability_20261021/`，
只用于工程验收，不据像素诊断宣称编辑成功率提升。

合并前已重新检查完整提交范围、资产、依赖与 CLI，结果见架构文档的合并检查记录；
GitHub PR 的合并状态单独确认。Stage 0 原始结果继续保留在独立 worktree。

`data/`、`models/`、`outputs/`、`external/`、`.venv/` 全部忽略。Git hooks 检查提交信息、代码行数和禁止上传的资产。实验原始日志应留在本地 `outputs/logs/`，一次性文件放 `.tmp/` 并在完成后清理。

所有 GPU 任务使用 `image-edit gpu -- ...`：实时检查占用、利用率和计算进程，再获取 UUID 锁。可用 `--device 2` 指定物理卡，若忙则拒绝；不允许通过直接设置 CUDA 序号绕过检查。锁协调同一用户的本项目任务，共享机器仍需遵循管理员调度。

评估补强新增 `prepare-controls --kind constant_mask`，按 `[source, 全黑RGB]` 提供
与Mask相同的双参考结构。`infer --reference-preprocessing training` 显式对齐
单参考的训练器cast顺序，默认 `auto` 保留旧No-mask。新结果必须使用新输出目录。

人工评估使用本地盲评包（至少两个同预算完整运行）：

```bash
image-edit review export --data-root <data> --manifest <dev.csv> \
  --run no_mask=<run-a> --run static=<run-b> --output <new-review>
# 本地打开 new-review/index.html，由真实评审者填写并下载CSV。
image-edit review score --bundle <new-review> --ratings <ratings.csv> --output <new-summary.json>
```

评分含编辑成功度、无关内容保留与失败类别/说明。空白为pending，不产生人工结论。

三位评审者分别填写（代号仅为组织占位，不能当作已收到评分）：

```bash
image-edit review panels --bundle <review> --reviewer rater_a --reviewer rater_b \
  --reviewer rater_c --output <new-public-panels>
image-edit review import --bundle <review> --ratings <real-rater-a.csv> \
  --reviewer rater_a --independent --output <new-import-a>
# 对其余真实评分分别 import，再合并；每位评审者只能选择一个版本。
image-edit review merge --bundle <review> --import <import-a> --import <import-b> \
  --import <import-c> --annotations <task-types.csv> --output <new-analysis>
```

原始CSV逐字节留存，三位完整独立评审者之前区间为null。按scene/session与reviewer
交叉bootstrap、固定seed、配对比较两项评分；NA不计零分。失败分类是结果筛选后的
描述，不作收益推断。工具无法证明真人独立性，须由评审组织者保证。

`review audit` 接受同名 `--run METHOD=PATH` 和 `--training METHOD=REPORT`、共同
`--data-root/--manifest`、`--annotations` 与新 `--output`，检查实际训练预算/资产，
并从真实图片重算按任务分组的像素诊断。固定30例任务描述在
`configs/experiments/stage1_task_groups.csv`；它不是成功/失败评分。

`image-edit gpu -- python -m src.inference.reference_smoke --data-root <data> --lora <lora> --output <new-smoke>`
可单独检查单/双参考结构与cast差异；这类smoke不代表编辑质量。

最小Blender数据（Python 3.11，可选安装 `pip install -e '.[blender]'`）：

```bash
image-edit scene generate --config configs/datasets/blender_translation.yaml --output <new-scene-data>
image-edit scene validate --root <new-scene-data>
# 固定抽样seed后显式展开；同seed/组数可先128，再512验证。
image-edit scene expand --seed 20261022 --train-groups 12 --dev-groups 4 --test-groups 4 \
  --resolution 512 --output <new-scenes.yaml>
image-edit scene generate --config <new-scenes.yaml> --output <new-expanded-data>
```

默认使用固定4.5.3的Cycles CPU，生成多强度平移、阴影开关反事实、空背景及对象占据
证据；只读校验从证据重算T/D/P并验证哈希、split、位移与背景补全。配置和原始证据
均留本地输出。当前范围是直接光硬阴影，不代表真实场景或复杂间接光的真值。

Oracle仅走显式实验接口：`src.datasets.scene_manifest.load_oracle_condition` 加载labels/valid，
`Flux2Backend.edit_oracle(..., allow_oracle=True)` 按 `[source, role_RGB]` 编码。
T/D/P为RGB单位向量，ignore为黑；普通 `edit`/`infer` 仍拒绝Oracle。
`image-edit gpu -- python -m src.inference.oracle_smoke --data-root <scenes> --output <new-smoke>`
验证真实条件消费与T/D置换敏感性；没有角色训练adapter时不能作为质量上界结果。

无系统写权限且导入bpy缺少X/GL库时，运行 `python -m src.infra.blender_runtime`。
该入口按固定URL/SHA256提取本地运行库；随后给 `scene generate` 添加
`--runtime-library-path .cache/blender/runtime/usr/lib/x86_64-linux-gnu`。
只影响渲染子进程，不修改系统库或其他任务的环境。

有效域分别排除旧、新物体轮廓（含占据并集内部的重叠边界）；生成失败或未完成的数据
不能通过常规验证，也不能加载为Oracle。失败目录保留，修复后使用新的输出目录。

盲评可以比较经过校验的独立数据副本；每个运行仍核对自身的控制来源，公开source/target
另按共同manifest验证。导出同时检查prediction的实际尺寸与RGB模式。

Blender可选依赖固定为bpy4.5.3、Cython3.3.0、zstandard0.25.0；沿用原命名Conda环境
的数值依赖锁。完整本机安装快照另存实验输出，不将环境目录上传。

`image-edit gallery --data-root <scenes> --manifest <scenes/dev/manifest.csv> --output <new-gallery> --show-roles`
会先校验场景，再展示RGB角色真值与独立有效域；T红、D绿、P蓝、ignore黑。
不加该选项时，原有No-mask/Mask画廊布局保持兼容。

当前场景beauty/反事实为32次采样、occupancy为16次；轮廓ignore与背景误差校验
处理抗锯齿差异。角色标签的可信范围以valid和Stage 2报告为准。

scene sweep支持物体初始位置、几何/颜色、点光位置/能量、背景棋盘和强度变化。
物理采样范围从 `configs/datasets/blender_variations.yaml` 读取，`scene expand` 可用
`--variations <yaml>` 选择；强度、相机与反事实策略仍在场景template中。默认范围
复现本轮v1物理参数，既有渲染/训练不重跑、不修改。
新test split按完整scene group保留；旧train/dev格式仍可读。Oracle/Dynamic Binary
用 `prepare-controls --kind oracle_role|dynamic_binary` 编译，前者one-hot角色，后者
valid内T∪D。二者都是依赖真值的上界，导出先做完整反事实校验，推理加载不读target。

Oracle LoRA 使用原训练入口，指定 `train-config --data-root <scenes> --control oracle_role
--control-root <roles>`；Static/Dilated/Dynamic Binary分别指定其kind。重载上界使用
`infer --data-root <scenes> --manifest <scenes>/dev/manifest.csv --control oracle_role
--control-root <roles> --allow-oracle --lora <checkpoint> --output <new-run>`。
训练端检查one-hot角色通道与实际batch，沿用固定ai-toolkit的flow-matching目标。

有界四组研究入口（激活Conda后，在冻结的已安装worktree中运行）：

```bash
python -m src.training.role_study prepare --data <verified-512-scenes> --output <new-study>
python -m src.training.role_study run --output <new-study> --phase smoke
image-edit gpu -- python -m src.inference.oracle_smoke --data-root <new-study>/smoke_oracle_role_42/data \
  --lora <smoke-checkpoint> --output <new-study>/oracle_reload_smoke
python -m src.training.role_study run --output <new-study> --phase capacity
python -m src.training.role_study run --output <new-study> --phase train --parallel 4
python -m src.training.role_study run --output <new-study> --phase infer --parallel 4
python -m src.evaluation.role_metrics --study <new-study> --output <new-diagnostics>
```

训练预算唯一配置为 `configs/experiments/role_validation.yaml`；各子任务均由GPU租约
入口启动，重复目录会失败。先单卡测量，再扩大batch/并行；同一对照不自动调batch。
准备中的计划只允许已就绪pilot，完整方法×seed矩阵准备齐全后才允许正式比较。
P区域报告source漂移，T/D报告target误差，分别按scene聚合，不称为编辑成功率。
评估绑定实际训练checkpoint摘要，并检查预测RGB与原尺寸；错误产物不能靠resize通过。
同时报告不编辑source的T/D误差参考，防止把少编辑误读成方法收益。
