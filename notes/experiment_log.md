# 实验记录

时间以 Asia/Shanghai 为准。每条记录说明配置、输入、产物、实际结果与限制。

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
