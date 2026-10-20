# Image Editing Research

以 **FLUX.2 Klein Base 4B** 建立可复现编辑与 LoRA 基线，随后研究随编辑强度变化的空间角色预测。方法改进尚未验证。

- [方案评审](docs/proposal_review.md)：可行性、贡献边界、标签定义与必要对照。
- [Stage 0 设计](docs/stage0_design.md)：当前准备工作的范围和验收。
- [原始 roadmap](Image_Editing_Preparation_and_InterEdit_Roadmap.md)：研究输入与长期路线。
- [Inter-Edit 代码走读](notes/inter_edit_code_review.md)：CJT 张量流及 FLUX 适配边界。
- [开发规则](AGENTS.md)：Conventional Commits、500 行限制、文档同步和 GPU 规范。

## 环境与资产

从项目根目录执行，Python 3.11。服务器驱动为 535，固定 PyTorch 2.7.1 / CUDA 12.6 的兼容组合，实际运行状态以实验记录为准。

```bash
bash scripts/setup_environment.sh
export PATH="$PWD/.venv/bin:$PATH"
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

MagicBrush 使用官方 train/dev、每个会话的第一轮，固定上游顺序；不接触 test。保持原尺寸，训练框架负责 bucket，推理按比例调整为 16 倍数。导出保存样本身份、指令、尺寸及 SHA256，严格检查配对和跨 split 泄漏。重复执行仅复用通过校验且 revision/数量一致的完整 split。

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

## 开发与目录

```bash
ruff check .
ruff format --check .
pytest -q
git add <本次改动>
git commit -m "feat(scope): concise description"
git push
```

`src/image_editing/` 放公共实现，`scripts/` 放环境和上游训练入口，`configs/` 放可审查配方；`docs/` 和 `notes/` 跟踪设计与实验。
`data/`、`models/`、`outputs/`、`external/`、`.venv/` 全部忽略。Git hooks 检查提交信息、代码行数和禁止上传的资产。实验原始日志应留在本地 `outputs/logs/`，一次性文件放 `.tmp/` 并在完成后清理。

所有 GPU 任务使用 `image-edit gpu -- ...`：实时检查占用、利用率和计算进程，再获取 UUID 锁。可用 `--device 2` 指定物理卡，若忙则拒绝；不允许通过直接设置 CUDA 序号绕过检查。锁协调同一用户的本项目任务，共享机器仍需遵循管理员调度。
