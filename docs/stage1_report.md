# Stage 1 实际运行记录

更新时间：2026-10-21。联合条件代码已实现；最小 GPU 训练、编码/消融已完成。
batch 4 容量检查已完成，三组 150-step 正式初步对照尚未开始。
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
BF16 下为 0.007261/0.021445，mask 最大绝对差 0.275391。两套实现的归一化数值
存在舍入差异，不能声称编码逐字节相等；未放宽预先规定的 0.001/0.05 RMSE 门限。
两边打包形状均为 `[1,2048,128]`，参考位置 ID 完全相同，T=10/20 各 1024 tokens。

所有 GPU 任务均先检查 `nvidia-smi`，通过 UUID 租约启动并正常释放。没有真实 OOM。

随后独立 `smoke/capacity_report.json` 完成 2 steps/batch4：192.63s、峰值 reserved
17.09 GiB，双图检查数为2，checkpoint有限。正式比较保持batch4，不自动放大。
为避免 Dilated 条件复核占用租约时间，方形膨胀优化为等价积分图：5张真实512²
mask逐像素相同，CPU单图均值0.583s→0.00883s，独立PIL回归覆盖半径1/2/16与边界。

## CPU 与待完成项

完整环境当前 **125 passed**，Ruff 通过。此前两次远端 CPU CI 因新增 fake backend
测试没有 mock package version 查询，缺 Torch 元数据而失败；已按原 CPU fixture
方式隔离版本查询，后续远端 checks 需重新确认，不能将本机通过写成远端通过。

三组相同 200/30、150 steps/batch4、seed42、固定 12 dev/50 steps/CFG4 对照尚未
运行。No-mask 只会在配置、checkpoint、实际数据和预算复核后复用；逐组结果、
真实耗时、峰值、失败及局限在完成后更新，不预填研究结论。
