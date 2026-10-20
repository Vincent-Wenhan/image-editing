# 项目开发规则

- 每个完整、可验证的改动都按 Conventional Commits 提交，如 `feat(data): ...`、`fix(infer): ...`、`docs(research): ...`。禁止将不相关改动混在一个提交中。用户要求推送时，正常 push，禁止默认 force push。
- 先记录设计、范围与验收方式，再实现；提交时同步 README、设计文档和 `notes/experiment_log.md`。明确区分计划、已实现、已运行和研究结论。
- 代码保持简洁，按职责拆分；每个自有代码文件不超过 500 行。复用公共逻辑，删除被替代实现，不保留重复版本和一次性脚本。
- `src` 是 Python package，自有模块使用 `src.xxx` 绝对导入；不得修改 `sys.path`。脚本只做入口，业务按 `docs/architecture.md` 的职责放入模块。推理输入与 target/oracle 监督分离；尚未实现的空间条件必须明确拒绝，不能静默忽略。
- 临时文件放 `.tmp/` 并在完成后清理；实验日志、配置快照和结果放 `outputs/`，它们是可复现实验资产，不是临时脚本。
- 数据、模型、生成图片、checkpoint、第三方仓库、环境、缓存和凭据禁止上传 GitHub；只提交下载入口、锁定 revision、配置及文字报告。提交前检查 staged 文件和体积。
- GPU 任务启动前必须实时查询 `nvidia-smi`：确认显存占用、利用率、计算进程；仅使用空闲卡，显式设置 `CUDA_VISIBLE_DEVICES`。不得中断他人任务。
- 使用 `image-edit gpu -- ...` 启动 GPU 工作，采用 UUID 锁避免本项目重复分配。单卡先测吞吐与峰值，再增加 batch/并行任务；优先模型常驻 GPU，尽量提高有效显存和算力使用率，保留 OOM 余量，禁止用无效分配“填满显存”。
- 固定模型与数据 revision、随机种子、依赖、配置和样本 ID；评估保持相同数据、采样与计算预算。官方 train/dev/test 严格分离，按会话检查泄漏。
- 首阶段固定未蒸馏 FLUX.2 Klein Base 4B。MagicBrush mask 不是真值角色场，像素差异不是必要物理后果标签；smoke test 不能证明性能提升。
- 外部源码仅放忽略目录 `external/` 并固定 commit；不随意修改外部源码。必要补丁必须版本化、解释原因和验证。
- 运行与改动相称的检查：格式/静态检查、关键数据和 GPU 调度逻辑测试、真实最小推理/训练。通过后不无故重复运行昂贵测试。
- 新环境执行 `git config core.hooksPath .githooks` 启用提交信息与仓库卫生检查。
