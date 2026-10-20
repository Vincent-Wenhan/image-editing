# Stage 0 状态与复现入口

更新时间：2026-10-21（Asia/Shanghai）。**稳定性回归 97 项通过，20-step debug、150-step Edit LoRA、固定 12 例重载、配对评估与画廊均已完成。验收状态为 complete；没有方法性能提升结论。**

## 本轮验收条件

代码位于独立 worktree `/home/jiangwenhan/project/image-editing-architecture`，分支
`refactor/modular-architecture`，执行代码 commit `26bc4c2`。未改主要目录、网络结构、
flow loss 或第三方源码。新运行使用：

```text
outputs/stage0_stability_20261021/
  status.json                 # 阶段、时长、尝试和错误
  environment.json            # 软件、GPU 与源码版本
  data_snapshot.json          # 原始 manifest SHA256 与副本验证
  data/                       # 200 train /30 dev 文件副本，隔离训练缓存
  base/                       # 复用的旧 12 例及重新计算的诊断
  debug.yaml, debug_report.json
  smoke.yaml, smoke_report.json, smoke_report.effective.yaml, smoke_report.memory.json
  lora/                       # 本轮 checkpoint；不上传
  lora_inference/              # 完成训练后生成
```

| 项目 | 固定条件 |
|---|---|
| 模型 | 未蒸馏 `black-forest-labs/FLUX.2-klein-base-4B` |
| 模型 revision | `a3b4f4849157f664bdbc776fd7453c2783562f4d` |
| ai-toolkit | `a055947d56a156411abb4ef0301b35d0fe5f51a5` |
| 数据 | MagicBrush `6c254560879df7851233ea0fd2ae2a62764d0ef2`，官方 first-turn，200 train /30 dev |
| 训练 | 512²、BF16、rank/alpha 16、AdamW、lr 1e-4、flowmatch/sigmoid、gradient checkpointing、seed 42 |
| 固定推理 | dev 前 12 例，512²、50 steps、CFG 4，按 sample ID 派生 seed |
| 环境 | Python 3.11.17，Torch 2.6.0+cu118，Diffusers 0.37.0，Transformers 4.57.3 |
| GPU | 空闲 RTX A6000，经 `image-edit gpu -- ...` 获取 UUID 租约 |

原模型权重未移动或重复下载。旧工作区数据只读验证后，逐文件复制到本轮 `data/`；
两个 manifest 的 SHA256 与原文件完全一致，session/source overlap 均为 0。
RGBA mask 的反向 alpha、几何和哈希规则保持不变。

## 已验证和运行中

| 项目 | 状态 | 本轮证据 |
|---|---|---|
| Ruff /97 项 Pytest /CLI | 通过 | 本地检查；含幂等、loss/梯度/更新一致性、异常清理、字节哈希和容量约束回归 |
| 依赖与版本 | 通过 | `pip check`、`environment.json`，外部 Git checkout 无修改 |
| 200/30 数据 | 通过 | `data_snapshot.json`、`data/validation.json` |
| Base 12 例 | 旧结果复用，重新校验与评估通过 | `base_reused_from`、`base/run.json`、`base/metrics.json` |
| 20-step debug，batch 2 | 完成，160 个 LoRA 张量均有限 | `debug_report.json`、`lora/debug/debug.safetensors` |
| 150-step Edit LoRA，batch 4 | 完成，160 个 LoRA 张量均有限 | `smoke_report.json`、`lora/smoke/smoke.safetensors` |
| 固定 12 例 LoRA 重载与配对评估 | 完成，配置/数据一致性检查通过 | `lora_inference/run.json`、`metrics.json` |
| 本轮最终核验 | 完成；checkpoint 步数/SHA、数据哈希、有限值和 12 例覆盖通过 | `verification.json` |
| Static Mask /Role /Qwen | 未实现，不属本轮 | 研究路线与架构文档 |

20-step debug 的真实报告耗时为 **199.58s**（含初始化、加载、缓存和清理），
峰值 allocated **15.02 GiB**、reserved **15.04 GiB**；checkpoint 元数据
`training_info.step=20`，完整文件 SHA256 为
`138ba8e4ca9ec9bef29199ca55146f883396df5f6c1d0e7d5c392424d43ac2b1`。
报告没有 `cleanup_errors` 或 `telemetry_errors`。

150-step Edit LoRA 的真实报告耗时为 **787.24s**（同样含初始化、加载和清理），
峰值 allocated **15.02 GiB**、reserved **15.10 GiB**；checkpoint 元数据
`training_info.step=150`、epoch=2，SHA256 为
`911bb69a934768805c942a635f50c3d13c6bea1717706d82cdd9e9b745fe4b31`。
没有实际 OOM、训练失败、清理或统计错误，未执行降级重试。

150-step 的计划在实际租约内计算：可用 47.99 GiB，保守估计 batch 4 峰值
30.07 GiB，预留 9.60 GiB，允许从已测 batch 2 增长到 4。
这是保守容量规划，实际峰值以上述运行报告为准，不是无 OOM 保证。
自适应只用于 Stage 0 工程验收；正式研究须固定有效 batch、训练样本和计算预算。

固定 12 例重载的逐图生成时间合计 **317.94s**（均值 26.50s）；包含 Python/模型
加载和输出的整个重载阶段为 **390.92s**。峰值 allocated **15.59 GiB**、reserved
**16.26 GiB**。记录中的 LoRA SHA256 与最终 150-step checkpoint 一致，12 张输出
均与 Base 不同。整个自动验收为 **1460.10s**（24 分 20 秒，不含此前的 CPU 测试及数据复制）。

验收结束后实时查询 8 张 GPU 均为 0 MiB /0%，无计算进程，租约正常释放。
原数据与独立副本在训练后再次验证，配对/字节哈希均通过，跨 split session/source
交集仍为 0；两个训练报告均没有清理/统计错误。没有实际失败或 OOM 降级尝试。

## 同配置 12 例像素诊断

Base 为旧记录复用，LoRA 为本轮重载；两者使用相同 manifest、样本顺序、模型
revision、seed、50 steps、CFG 4、512²，评估 threshold 均为 0.05。

| 诊断 | Base | Edit LoRA 150-step |
|---|---:|---:|
| Target MAE | 0.073098 | 0.055915 |
| Identity→target MAE | 0.030950 | 0.030950 |
| 变化面积占比 | 0.302519 | 0.188654 |
| Source drift MAE | 0.069352 | 0.052436 |
| Mask 外漂移 proxy | 0.054625 | 0.034876 |

误差/漂移较低，同时变化面积也下降；identity 的 target MAE 仍更低，说明这些
像素指标不能代替指令完成度或必要后果评价。单训练 seed、12 个 dev 样本和短训练
仅证明训练、保存、重载及评估链路可用，不证明编辑成功率提升或方法优越性。
画廊为 `base_gallery/index.html` 与 `lora_gallery/index.html`，各有两张 contact sheet。

## 稳定性修复及兼容边界

- 设备补丁绑定固定上游版本，重复安装不叠加；只移动 VAE device，保留 dtype 与原调用。
  小型 CPU 网络验证了 loss、梯度和两次 AdamW 更新一致，不声称未补丁 4B 模型的 CUDA 数值对照。
- worker 在失败路径尝试 cleanup、显存统计和缓存释放，次级异常不遮蔽原异常/traceback；
  launcher 等待其退出。失败/取消/统计与清理失败场景通过单元测试。
- 评估先复核 manifest 和 source/target/mask 的实际 SHA256、模式、尺寸及样本身份。
  新运行有可选 `config_sha256`；旧记录仍可读。配对评估通过 `--reference-run` 比较
  模型版本、采样预算、seed 和样本集合。指标公式及原 CSV/JSON 指标键保持不变。
- EditInput、EditSupervision 与 ControlBundle 分离。RGB、同尺寸 mask、二值有效域、
  硬标签与软 T/D/P 概率场约束见 [architecture.md](architecture.md)；当前后端拒绝尚未实现的控制。
- batch 按实际容量与 debug 峰值保守规划，最多 2 倍增长，预留 max(20% 容量, 4 GiB)。
  放大后若 OOM，只以新实验名降回已测 batch 重试一次；其他异常直接失败。
  OOM 降级仅通过 CPU 注入测试验证，本轮没有触发，不宣称完成真实 OOM 恢复试验。

## 复现与历史记录

从 worktree 根运行，保留原命名 Conda 环境的安装状态：

```bash
conda run --no-capture-output -n image-edit "$PWD/.venv/bin/python" scripts/run_stage0.py \
  --data-root "$PWD/outputs/stage0_stability_20261021/data" \
  --base-run /home/jiangwenhan/project/image-editing/outputs/stage0/base \
  --output outputs/stage0_new_run
```

新实验必须使用新输出目录；脚本不静默覆盖旧 status、训练配置或报告。
原工作区 `outputs/stage0/debug_report.json` 的第一步 CPU/CUDA 拼接失败原样保留；
`outputs/background_jobs.json` 是历史启动清单，不代表当前活动进程。
上一轮架构迁移的 50 测试、单例 Base 字节一致及两步 LoRA/单例重载证据仍在
`outputs/refactor_validation/`。本轮 Base 只复用旧记录，没有再次计入 GPU 推理耗时。

完整环境由 [environment.yml](../environment.yml)、[Conda lock](../configs/conda-linux-64.lock)
和 [pip lock](../requirements.lock) 恢复。模型、数据、checkpoint、图片及原始日志不上传 GitHub。
短训练和像素诊断用于工程验收，不代表编辑成功率、物理因果准确率或优于基线。

## 尚未完成的研究工作

Static Mask、Role Predictor、Blender/Kubric 与 Qwen 训练均未实现，也未在本轮运行。
下一阶段先人工审查当前固定样本及失败案例，定义编辑完成度/保护性评价和公平预算，
再按研究方案进入 Static Mask；不能把本轮较低像素漂移当作研究 Go 结论。
