# Stage 0 状态与复现入口

更新时间：2026-10-21（Asia/Shanghai）。**200/30 数据与完整 Base 权重已准备，旧流程 12 例 Base 推理完成；LoRA debug 在第一步失败，尚无完整训练验收。架构重构已完成模块迁移，正在进行兼容性验收。**

## 已完成

- 新建独立 Conda 环境 `image-edit`：`/gluster/envs/users/jiangwenhan/conda-envs/image-edit`。
- Python 3.11.17、PyTorch 2.6.0+cu118、Diffusers 0.37.0、Transformers 4.57.3、PEFT 0.18.1。环境内有 C/C++ 编译器；`pip check` 通过。
- 已在检查为空闲的 RTX A6000 上验证 BF16 matmul、Klein pipeline 导入和 ai-toolkit `ExtensionJob` 配置构建。配置构建不等于 forward/backward 已完成。
- `data/magicbrush_smoke/`：89 train /30 dev first-turn 样本，两个集合的 session 和 source 图像 SHA256 交集均为 0；图像、指令、尺寸、Mask 与校验和检查通过。
- 实测并修正 RGBA mask 反向 alpha 语义及 500²→512²、1024²→512² 的配对尺度转换；已检查 30 组 source/target/mask 画廊。
- 重构前 19 项 CPU 测试通过；迁移并补充协议、配置、CLI 和记录测试后 42 项通过。
- 独立 worktree 内 `pip install -e .`（复用锁定依赖）和 `pip check` 通过；从其他目录导入及九个子命令帮助均通过，帮助和模块导入不加载 torch/CUDA。
- 新结构下 1 例真实 Base 推理通过，512² /50 steps /CFG 4，与旧样本 PNG SHA256 完全一致；原 12 例结果保留。
- 已固定外部代码/model/dataset revision，建立 Conventional Commits hooks、仓库资产检查与 CPU CI。

## 尚未完成的验收

| 项目 | 当前状态 | 证据/入口 |
|---|---|---|
| 200 train /30 dev 正式准备子集 | 已完成 | `data/magicbrush/dataset.json` |
| Base 全部权重和本地文本编码器链接 | 已完成；复用已有资产 | `models/FLUX.2-klein-base-4B/download.json` |
| 固定 12 例 Base 编辑推理 | 原流程已完成 | `outputs/stage0/base/run.json` |
| 20-step debug、150-step LoRA、同配置重载 | debug 在第一步因 VAE 条件 CPU/CUDA 混用失败，后续未执行 | `outputs/stage0/debug_report.json` |
| 架构重构验收 | 原 42 测试和 6 个设备适配测试通过；1 例 Base、两步训练及 1 例 LoRA 重载通过，最终安装/兼容审计待汇总 | [架构文档](architecture.md) |
| 静态 Mask、动态 Role、Qwen 迁移 | 研究后续阶段，尚未实现 | 原始 roadmap 与 proposal review |

网络存在 SSL 超时、连接重置和大文件 CDN 签名过期；传输使用可续传的 4 MiB 分块，逐次通过稳定 URL 获取重定向，最终验证固定版本的完整文件 SHA256。下载分块数量和下载进程存在都不代表模型已验收。

旧后台任务已经结束，`outputs/background_jobs.json` 是历史启动清单，不能作为当前进程状态。
原验收状态见 `outputs/stage0/status.json`，失败记录原样保留。
重构验证在独立 worktree 的 `outputs/refactor_validation/` 写入新记录。

## 使用

```bash
conda activate image-edit
image-edit validate --root data/magicbrush_smoke
image-edit gpu -- image-edit env --cuda
```

环境从 [environment.yml](../environment.yml)、[Conda lock](../configs/conda-linux-64.lock) 和 [pip lock](../requirements.lock) 恢复，或执行 `bash scripts/setup_environment.sh`。先建立 Conda 环境，再应用 pip lock；后者固定训练器需要的 setuptools 69.5.1。PyAV 是上游模型注册表的导入依赖，本项目仍只训练图像模型。

资产恢复入口：

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/prepare_assets.sh models
HF_ENDPOINT=https://hf-mirror.com bash scripts/prepare_assets.sh data
python scripts/run_stage0.py --output outputs/stage0_new_run
```

验收流程最多等待资产 24 小时，随后执行 Base 推理、诊断、20-step/150-step LoRA 和配对重载，记录逐步日志与状态。debug 使用 batch 2；若其峰值 reserved 显存低于 22 GiB，再用 batch 4 做 150-step 检查。任何 GPU 工作都通过空闲检查与 UUID 锁启动；等待下载时不占用 GPU。失败即记录并停止，不将失败写成成功。

画廊：`outputs/data_audit/gallery/index.html`、`outputs/dev_gallery/index.html`。所有数据、图片、权重、检查点和原始日志均留在服务器，不上传 GitHub。

## 研究判断

路线可执行，但贡献应聚焦“随强度预测必要连带变化”，并控制标签泄漏和比较预算。[方案评审](proposal_review.md) 与 [Inter-Edit 走读](../notes/inter_edit_code_review.md) 给出具体修正和对照要求。现有测试与未来短训练只证明工程路径；尚无方法优于基线的研究结论。
