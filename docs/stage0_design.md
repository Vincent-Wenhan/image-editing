# Stage 0 实施设计

状态：实施中。日期：2026-10-20（Asia/Shanghai）。

## 范围与验收

1. 审阅原始 proposal，核对官方模型、训练器、Inter-Edit 和直接相关工作，记录方法定义与公平比较的缺口。
2. 固定外部源码、FLUX.2 Klein Base 4B 和 MagicBrush revision；下载 200 train / 30 dev 首轮编辑样本，不下载 test、Qwen 或全量 Inter-Edit。
3. 建立独立 Python 3.11 环境，对齐服务器 535 驱动；验证 CUDA、依赖和本地离线模型加载。
4. 建立可安装 Python 包及统一 CLI，覆盖资产下载、数据转换/验证、GPU 空闲检查与锁、推理、基础诊断及 ai-toolkit 训练配置。
5. 实测固定 dev 样本原始编辑，完成最小 LoRA 训练和重载验证；若外部条件阻断，记录真实错误与可恢复命令，不将准备好的配置写成运行成功。
6. 执行代码检查、关键逻辑测试和仓库卫生检查，分逻辑 Conventional Commits 提交并推送。

## 数据契约

- `data/magicbrush/{train,dev}/{reference,target,mask}`，target 中同名 `.txt` 为指令。
- manifest 保留原始 session、turn、split、尺寸、文件路径与 SHA256；仅按固定 revision 顺序选取 first-turn 样本。
- 实测源图 500×500、目标/Mask 1024×1024；核对宽高比后统一到长边 512 的 16 倍数尺寸，保留原尺寸记录；禁止独立裁剪。Mask 使用最近邻，不使用 kit 原来的 Lanczos。宽高比不一致必须失败。
- 实测 RGBA mask 的透明区为编辑区域：取反 alpha 并在 128 阈值二值化；普通灰度 mask 才按白色为编辑区处理。导出白色为允许编辑区域，记录实际编码。缺文件、重复 ID、空指令和跨 split 会话泄漏必须失败。
- 输出目录不可静默覆盖；不把不足指定数量或缺少预测的评估报告为完整通过。

## 推理与训练

- 原始推理用 `Flux2KleinPipeline`、BF16、50 steps、CFG 4.0，模型常驻空闲 GPU。
- 每次运行保存固定样本、种子、模型 revision、软件版本、配置、耗时与峰值显存；稳定种子由 sample ID 派生，不能依赖 batch 顺序。
- ai-toolkit 使用同名 source/target + `control_path`；适配固定 commit 的真实字段；保留原始 source、target、mask 供后续静态控制。
- 先用短训练测显存与有效 batch，再运行 100–200 steps 的工程验证；保存 LoRA 并用同一 dev 配置重载比较。
- 空闲卡检查与 UUID 文件锁贯穿任务生命周期；锁只能协调本项目，不能代替共享服务器的全局调度。

## 文档与资产

- 原始 roadmap 保留作为输入；评审结论放 `docs/proposal_review.md`，实际状态以 `notes/experiment_log.md` 和 `docs/stage0_report.md` 为准。
- `configs/assets.lock.json` 记录下载源与 revision；环境完整冻结到 requirements lock。
- `outputs/` 保存可复核原始记录和画廊，但所有数据和权重均不进 Git。
