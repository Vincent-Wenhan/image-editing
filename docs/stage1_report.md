# Stage 1 实际运行记录

更新时间：2026-10-21。**工程验收 complete：最小 GPU 验证、三组150-step训练、
固定12例重载/配对评估和画廊均已完成。尚无优于No-mask的研究结论。**
架构已通过 [PR #1](https://github.com/Vincent-Wenhan/image-editing/pull/1) 合并到 main；
Stage 0 全部结果保留，当前开发位于 `feat/static-spatial-control` 独立 worktree。

## 固定条件与资产

- 沿用 Stage 0 的 FLUX.2 Klein Base 4B、模型/数据 revision、依赖与 ai-toolkit commit。
- 新目录 `outputs/stage1_20261021/`；原权重通过忽略链接复用，未重下或移动。
- `smoke/data` 是原 200 train /30 dev 的文件副本，manifest 字节不变，配对/哈希验证通过。
- Static/Dilated/Shuffled 的定义、图序、数值及公平预算见 [设计](stage1_design.md)。
- No-mask 默认仍走原 Diffusers 单 source 路径；Mask 才使用匹配训练 cast 顺序的双参考路径。

## 已运行的最小验证

首次 CLI 配置写出遇到 StrEnum/YAML 序列化错误，后续训练在缺配置预检退出，
没有模型加载或训练 step。失败日志与 `preflight_failure.json` 均保留；修复为普通
字符串并补充 YAML 写出/重读回归，没有覆盖原失败记录。

`smoke/train_02_report.json`：2 steps、batch1、rank/alpha16、seed42、512²、原 flow
loss。执行代码 `272f82e`，实际双参考 batch 检查为 2；160 个张量有限。
worker 耗时 **194.61s**（含初始化、加载、200 例缓存与清理），峰值 reserved
**15.04 GiB**。checkpoint 为 `smoke/lora/joint_debug_02/joint_debug_02.safetensors`，
SHA256 `c048974ffac3c57daa5ec2a9bd2bb8c3cc1c9637a75b012b5ab65af4a63d6295`。

`smoke/ablation/report.json`：执行代码 `3ddaf92`，重载上述 checkpoint，固定
`dev_140513_t1`、seed 3662387196、512²、4 steps、CFG4。四次调用共用常驻模型：
Static、全零 Mask、Shuffled、Static repeat。每次真实 VAE encoder 恰好接收两图，
mask 输入只含 -1/+1；attention 的 T=0/10/20 各有 1024 tokens。

Static/零图/Shuffled 的 source token 和初始 noise 摘要相同，mask token 摘要不同，
生成 PNG 也不同。Static repeat 的 PNG SHA256 与第一次完全相同：
`a60418c097758dde0692441a40a7f39dc1077bf5457cbbd881a9bc578a701076`。
这证明条件被消费和采样可重复，不证明模型正确理解 mask 或编辑质量提高。

编码/消融 worker **110.64s**，包含启动器的 wall **126.63s**；四次生成分别为
3.34/3.21/3.22/3.22s，峰值 reserved **16.56 GiB**。真实 VAE 采用相同 BF16
权重量化后比较：FP32 算术下 source/mask 相对 RMSE 为 0.000306/0.000414；
BF16 下为 0.007261/0.021445，mask 最大绝对差 0.275391。两套实现存在数值差异，
未单独定位各算子的舍入贡献，不能声称编码逐字节相等；未放宽预先规定的
0.001/0.05 RMSE 门限。
两边打包形状均为 `[1,2048,128]`，参考位置 ID 完全相同，T=10/20 各 1024 tokens。

所有 GPU 任务均先检查 `nvidia-smi`，通过 UUID 租约启动并正常释放。没有真实 OOM。

随后独立 `smoke/capacity_report.json` 完成 2 steps/batch4：192.63s、峰值 reserved
17.09 GiB，双图检查数为2，checkpoint有限。正式比较保持batch4，不自动放大。
为避免 Dilated 条件复核占用租约时间，方形膨胀优化为等价积分图：5张真实512²
mask逐像素相同，CPU单图均值0.583s→0.00883s，独立PIL回归覆盖半径1/2/16与边界。

## 三组150-step训练及checkpoint

执行代码 `46c5885`，200 train/30 dev、512²、batch4、rank/alpha16、seed42，
AdamW/lr1e-4、flowmatch/sigmoid、BF16，与Stage 0 No-mask完整配方/预算一致。
每组从同一Base独立开始，没有续训debug/capacity adapter。Static/GPU2、
Dilated/GPU0、Shuffled/GPU1使用互异UUID租约，并行执行。

| 方法 | worker耗时(s) | 含启动器wall(s) | 峰值allocated/reserved(GiB) |
|---|---:|---:|---:|
| No-mask（历史复用） | 787.24 | 见Stage 0报告 | 15.02 /15.10 |
| Static | 1234.75 | 1257.89 | 15.02 /17.09 |
| Dilated | 1230.83 | 1254.26 | 15.02 /17.09 |
| Shuffled | 1232.95 | 1256.04 | 15.02 /17.09 |

三组report均complete，实际双控制检查数均150，checkpoint的160个张量均有限，
header均为step=150/epoch=2。没有OOM、训练失败、cleanup或telemetry错误。
最终权重位于 `comparison/<kind>/lora/<kind>/<kind>.safetensors`：

| kind | checkpoint SHA256 |
|---|---|
| static_mask | `e2d4e2e6dcbbae7bc40860c34af6a41d4aa957fe9b4cd7a36c1be7c3f299bbc6` |
| dilated_mask | `e8ed862e2a46095513221648366884bb7191292b5a27ac9b488d02dad44fc095` |
| shuffled_mask | `461b9a621fb5136466d4bfa2c4ac05f7c5b7eaf99655fa6c30a0018540dff77a` |

## 固定12例重载与诊断

执行时git head为`41cd05e`（与训练的可执行代码相同，仅追加运行文档）。
各组保持相同manifest/样本顺序、稳定sample seed、50 steps、CFG4、512²。
No-mask为完成的Stage 0 **150-step Edit LoRA**，并非未训练Base。
复用前核实配置、checkpoint及实际数据SHA256，未重跑或覆盖旧结果。

| 方法 | 逐图生成合计(s) | 含加载/启动器的推理阶段(s) | 峰值reserved(GiB) |
|---|---:|---:|---:|
| No-mask（历史复用） | 317.94 | 390.92 | 16.26 |
| Static | 450.31 | 721.90 | 16.39 |
| Dilated | 456.73 | 728.81 | 16.39 |
| Shuffled | 450.22 | 721.61 | 16.39 |

含加载时间受到并行任务和CPU/存储影响，不能把时间比例直接等同算法FLOPs比例。
双参考增加token和计算；这里匹配更新/样本预算，不声称相同FLOPs。
三组都完成12例，评估预算/版本/哈希检查通过；阈值均0.05，评估区域统一使用原始
MagicBrush mask，不能按各方法变换后的mask改评估区域。

| 方法 | Target MAE | Source drift MAE | Mask外漂移proxy | 变化面积占比 |
|---|---:|---:|---:|---:|
| No-mask | 0.055915 | 0.052436 | 0.034876 | 0.188654 |
| Static | 0.057758 | 0.054282 | 0.035403 | 0.193137 |
| Dilated | 0.056720 | 0.052455 | 0.034180 | 0.186667 |
| Shuffled | 0.074145 | 0.070708 | 0.054294 | 0.234575 |

四组identity→target MAE均为0.030950。Static/Dilated没有降低No-mask的target MAE；
Dilated的外漂移略低，但变化面积也略少。Shuffled的像素误差/漂移较大，仅是这次小样本
诊断，不能据此宣称统计显著性、编辑成功率提高或方法优越性。

每组 `inference/metrics.csv` 保留逐例值；`gallery/` 保留初次四列画廊。
审查时发现原画廊mask列仅为标注，因此新增 `gallery_controls/`：source、target、
annotation_mask、control_input、prediction五列，实际控制图按运行尺寸重建。
No-mask原布局不变，初次画廊没有覆盖。目视抽查Static的`dev_319096_t1`镜面出现
白块残留，可能涉及控制图外观污染；这是探索性观察，未做盲评或确定机制归因。

## 150-step Static的同checkpoint消融

`comparison/static_mask/trained_ablation/` 重载最终Static权重，保持同一源图、指令、
seed3662387196、50 steps、CFG4，依次Static/零图/Shuffled/Static repeat。
worker425.88s，source/noise token相同、mask token及输出变化；重复Static逐字节相同。
该Static PNG还与常规12例推理中的同ID PNG完全一致，验证独立重载协议一致。

相对Static输出，零图/Shuffle的像素MAE分别0.029828/0.026414，变化面积分别
0.167091/0.109276；重复为0。这验证训练后仍对Mask敏感，不证明正确空间约束。

## 工程检查与限制

补齐实际控制图视图后，完整环境 **130 passed**（111.77s），Ruff、依赖与十个CLI/薄入口通过。
此前两次远端 CPU CI 因新增 fake backend
测试没有 mock package version 查询，缺 Torch 元数据而失败；已按原 CPU fixture
方式隔离版本查询；修复后远端 checks 已通过，完整日志保留在本地输出目录。
最新可执行代码的GitHub CPU CI为125 passed/5 skipped；五项需要Torch的测试在本机
完整环境均执行，真实GPU证据如上，不将CPU CI的跳过计作GPU通过。

最终 `outputs/stage1_20261021/verification.json` 核对原数据和三组副本、源/目标/mask
实际SHA256、split、条件索引、checkpoint、固定12例、采样预算、指标和画廊。
跨split会话/source交集均0；ai-toolkit、Inter-Edit及Diffusers三个Git checkout均为
固定commit且干净，Diffusers安装版本为0.37.0。
结束时8张GPU均0 MiB/0%，无计算进程，全部租约已释放。所有资产仅留本地。

当前只有一个训练seed、按固定顺序选择的12个dev样本和像素诊断；没有语义成功率、
保护性盲评或置信区间。为保持原baseline，No-mask使用原Diffusers cast顺序，Mask路径
对齐训练器cast顺序；这也是初步比较的数值差异因素，不能把差异全归因于空间信息。
后续先做语义/失败样例审查、保持旧baseline的匹配预处理对照、Mask扰动和多seed验证。
本轮不产生进入Dynamic Role的研究Go结论，未实现Role Predictor、Blender或Qwen训练。

## 使用与复现入口

本机从 `/home/jiangwenhan/project/image-editing-static-spatial-control` 运行，使用该
worktree的 `.venv`（复用命名Conda环境依赖）。训练/推理的完整argv已经写入每组
`postprocess_report.json`，预算以 `configs/experiments/stage1.yaml` 和原训练配方为准。
复现必须换输出目录，先用 `src.datasets.snapshot.snapshot_pairs` 复制验证过的数据，
再按README执行 `prepare-controls`、`train-config`、租约内训练与推理。

```bash
conda activate image-edit
source .venv/bin/activate
image-edit --help
# 使用已保存的配置重训时，先换name/training_folder/report，保留本轮checkpoint。
image-edit gpu -- python -m src.training <new-config.yaml> --report <new-report.json>
image-edit gpu -- image-edit infer --data-root <data> --manifest <dev-manifest> \
  --control static_mask --control-root <controls> --lora <checkpoint> --output <new-run>
image-edit evaluate --data-root <data> --manifest <dev-manifest> \
  --predictions <new-run> --reference-run <preserved-no-mask-run>
image-edit gallery --data-root <data> --manifest <dev-manifest> \
  --predictions <new-run> --output <new-gallery>
```

原始证据包括 `comparison/training_verification.json`、各组 `budget.json`、
`train_report.json`、`postprocess_report.json`、`inference/run.json`/`metrics.csv`、
`gallery_controls/index.html`，以及Static的 `trained_ablation/verification.json`。
所有模型、数据、checkpoint、预测与原始日志均不进入Git。
