# 工程架构重构

## 范围与基线

本次只重构 Stage 0 工程结构，保持 FLUX.2 Klein Base 4B 的采样、训练目标、
MagicBrush 预处理、CLI 参数和实验记录格式。Static/Dilated/Soft Mask、Role
Predictor、Blender/Kubric 与 Qwen 均为未来扩展，不在本次实现。

重构从 `5c505e5` 开始，在独立 `refactor/modular-architecture` worktree 中进行。
原工作区已有的 roadmap 移至 `docs/` 的改动内容完全一致，沿用该位置。
原工作区、命名 Conda 环境、第三方源码和所有资产保留；隔离测试环境只复用已安装依赖。

开始前已阅读 AGENTS、README、全部研究方案/报告/走读、现有代码、配置和测试。
原有 19 项 CPU 测试通过。实时检查没有运行中的项目任务，8 张 A6000 均空闲。
本地资产已完整下载，200 train /30 dev 已准备；旧 Stage 0 的 12 例 Base 推理完成，
debug 训练在第一步因 CPU/CUDA 条件张量混用失败，尚无完成的 LoRA 验收。
这些是重构前的状态，不是重构后的验证结果。

## 实施顺序与验收

1. 先记录设计、兼容边界和验收方案（本提交）。
2. 按职责拆分 core / datasets / conditioning / backends / training / inference /
   evaluation / infra；删除旧包，迁移 CLI、安装入口、脚本和原有测试。
3. 增加协议、配置、安装和旧产物兼容测试；运行 Ruff、Pytest、CLI smoke。
4. 复用本地资产，在 GPU 租约下做最小真实推理；训练验证单列真实结果及已知阻断。
5. 同步 README、Stage 0 报告、实验日志、旧新映射和最终检查结果，分阶段提交推送。

每一步提交前执行与改动相称的检查。新输出使用独立目录，不覆盖旧实验。

## 模块与依赖设计

`src` 是真正的 Python package，所有自有代码使用 `src.xxx` 绝对导入。
setuptools 从仓库根发现 `src` / `src.*`，console entry point 保持 `image-edit`；
模块入口变为 `python -m src`。不提供同名顶层 `datasets` 包，不修改 `sys.path`。

| 模块 | 职责 | 允许依赖 |
|---|---|---|
| `src/core` | 输入/监督契约、IO、配置、路径、种子、实验元数据 | 标准库、Pillow、YAML |
| `src/conditioning` | 控制图容器与现有 Mask 语义 | core、Pillow |
| `src/datasets` | MagicBrush 读取/对齐/导出、manifest、哈希与 split 验证 | core、conditioning、infra 下载 |
| `src/backends` | 公共编辑协议、FLUX.2 Diffusers 适配器 | core、conditioning、infra GPU |
| `src/inference` | 样本选择、后端调用、增量 run.json 和输出检查 | core、datasets、backends |
| `src/training` | 训练配置、固定 ai-toolkit 适配、checkpoint 验证 | core、datasets、infra |
| `src/evaluation` | 像素诊断、历史产物评估、画廊 | core、datasets、NumPy、Pillow |
| `src/infra` | 固定资产下载、断点续传、GPU 租约、环境及验收编排 | core；编排通过 CLI 调用业务 |
| `scripts` | 保留现有命令的轻量入口、环境 bootstrap、仓库检查 | 已安装的 src 包 |

底层数据对象不引用后端；推理流程不接收训练监督。重型 torch/diffusers/upstream
导入只发生在实际 GPU 执行时，帮助命令与 CPU 数据检查不初始化 CUDA。

## 数据与执行边界

- `EditInput` 只有 source、instruction、可选 mask/strength；不含 target、role label。
- `EditSupervision` 保存 target、可选角色标签和有效区域，单独显式加载。
- `ControlBundle` 描述 none / static / dilated / soft / oracle / predicted 控制图及元数据；
  容器不实现控制算法。Oracle 条件必须显式标记，普通推理拒绝它。
- `EditingBackend` 只统一编辑调用与输出；模型内部 Transformer 保持独立。
- 当前 FLUX 后端只支持无空间条件的 Stage 0；收到尚未实现的控制时明确报错，不能静默忽略。
- manifest 的推理读取路径只加载 source 和 instruction；训练与评估才读取 target。

配置保留 `configs/assets.lock.json` 的固定资产格式和现有完整 ai-toolkit YAML。
数据选择、实验/验收预算与模型资产分开；Python 和 shell 从配置读取默认值，
CLI 显式覆盖仍有效。生成的训练 YAML 仍可直接交给固定 ai-toolkit。
相对用户参数保持相对调用目录，仓库自带配置和第三方源码用明确的项目根定位。

ai-toolkit 在其根目录下以独立 Python worker 启动，使用正常的 `-m` 导入规则；
不改变父进程 cwd/导入路径，不修改第三方文件。GPU 租约、离线加载、commit
校验、checkpoint 有限性和现有 report 字段继续保留。

最终目录、运行指南、迁移映射及实测结果在实施后补齐。
