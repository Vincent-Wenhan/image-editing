# Image Editing Research

Stage 0 架构已通过 [PR #1](https://github.com/Vincent-Wenhan/image-editing/pull/1)
合并。Stage 1 的双参考条件、128 项回归和最小 GPU 编码/消融验证已完成，三组
150-step 初步对照运行中。见 [设计](docs/stage1_design.md) 和 [实际报告](docs/stage1_report.md)。

以 **FLUX.2 Klein Base 4B** 建立可复现编辑与 LoRA 基线，随后研究随编辑强度变化的空间角色预测。方法改进尚未验证。

- [方案评审](docs/proposal_review.md)：可行性、贡献边界、标签定义与必要对照。
- [Stage 0 设计](docs/stage0_design.md)：当前准备工作的范围和验收。
- [当前验收状态](docs/stage0_report.md)：已验证内容、下载进度入口和待运行实验。
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
