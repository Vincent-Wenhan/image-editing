# Image Editing Research

以 **FLUX.2 Klein Base 4B** 建立可复现编辑与 LoRA 基线，随后研究随编辑强度变化的空间角色预测。方法改进尚未验证。

- [方案评审](docs/proposal_review.md)：可行性、贡献边界、标签定义与必要对照。
- [Stage 0 设计](docs/stage0_design.md)：当前准备工作的范围和验收。
- [当前验收状态](docs/stage0_report.md)：已验证内容、下载进度入口和待运行实验。
- [原始 roadmap](docs/Image_Editing_Preparation_and_InterEdit_Roadmap.md)：研究输入与长期路线。
- [架构重构设计与验收](docs/architecture.md)：模块边界、兼容约定与实施步骤。
- 当前正在按架构文档收尾：补丁幂等、失败清理、评估校验和容量约束 batch；20/150-step 验收结果随后记录。
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
保留原 dtype、forward 和 loss。已通过两步真实训练、checkpoint 有限性检查与一例 LoRA 重载推理，
这不代表原定 20/150-step 全量基线已完成。

也可运行 `python scripts/run_stage0.py --output outputs/stage0_new_run`，等待固定资产准备完成后自动执行上述工程验收；状态写入该目录的 `status.json`。每次 GPU 启动仍会检查空闲卡，下载等待阶段不占用 GPU。新验收使用新的输出目录。

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
普通推理拒绝 oracle。当前后端仍只实现 Stage 0，无空间控制；Mask/Role/Qwen 的扩展位置已预留为接口。

本次重构验收：完整环境 50 项测试通过；纯 CPU 环境 49 项通过、1 项因未安装 torch 跳过。
editable/wheel 安装、九个子命令帮助、数据校验、旧 12 例评估均通过；一例 Base 输出与旧结果
逐字节一致，两步 LoRA 训练与一例重载完成。原定 20/150-step 完整基线仍待单独验收。

`data/`、`models/`、`outputs/`、`external/`、`.venv/` 全部忽略。Git hooks 检查提交信息、代码行数和禁止上传的资产。实验原始日志应留在本地 `outputs/logs/`，一次性文件放 `.tmp/` 并在完成后清理。

所有 GPU 任务使用 `image-edit gpu -- ...`：实时检查占用、利用率和计算进程，再获取 UUID 锁。可用 `--device 2` 指定物理卡，若忙则拒绝；不允许通过直接设置 CUDA 序号绕过检查。锁协调同一用户的本项目任务，共享机器仍需遵循管理员调度。
