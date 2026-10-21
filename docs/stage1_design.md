# Stage 1：静态空间联合条件

状态：工程验收及三组初步对照已完成，130项回归通过；尚未证明优于No-mask。
起点为 PR #1 正常合并后的 main `020c370`，
分支 `feat/static-spatial-control`。保留 Stage 0 目录和全部实验资产。

## 范围与依据

只实现 Static Mask、Dilated Mask、Shuffled Mask 的 `[source, mask_RGB]`
联合参考条件。No-mask 使用原单 source 路径和训练目标；不开发 Soft Mask、角色
预测、Blender 或 Qwen。MagicBrush 编辑区是空间提示，不是角色真值。

- [Inter-Edit CJT](https://github.com/Delong-liu-bupt/Inter-Edit/blob/220c1dc26d8401cae160a6b25e48198f85bdd115/train_qwen_edit_lora.py)
  按 `[source, mask]` 编码，连接 `[noisy_target, source, mask]`，只预测 target。
  FLUX 只迁移该概念，不复用 Qwen 的视觉语言 prompt 编码、网络或权重。
- [ai-toolkit FLUX.2](https://github.com/ostris/ai-toolkit/blob/a055947d56a156411abb4ef0301b35d0fe5f51a5/extensions_built_in/diffusion_models/flux2/flux2_model.py)
  支持 `control_path` 列表与逐样本原始多控制图。输入 RGB [0,1]，编码前转
  [-1,1]；逐图 VAE mean、2×2 latent patchify、running BN 后拼接。
- 固定 Diffusers 0.37.0 的 Klein pipeline 接收图像列表，使用相同的 mean/patchify/BN
  概念；位置 T=10/20 对应 source/mask，target T=0。实际编码数值必须实测。

上游会跳过缺失控制文件；非 full-size 路径会 bicubic resize。因此本项目必须在上游
前检查完整配对、实际哈希、顺序和尺寸，在调用边界验证两张参考。第三方源码不修改。

进一步核对发现 ai-toolkit 先将 [0,1] RGB cast 为 BF16 再归一化，Diffusers 默认先
归一化再 cast，两者会有量化差异。双参考推理采用限定作用域的原始参考预处理，
与训练 cast 顺序一致；结束/异常时恢复处理器。No-mask 保留原 Diffusers 路径。

## 数据与条件协议

方形二值膨胀使用 NumPy 积分图的矩形窗口求和，语义与 PIL MaxFilter 完全一致；
加入随机图/边界/不同半径的 PIL 对照，避免大核逐像素排序拖慢每次条件校验。

1. 固定 MagicBrush revision、200 train /30 dev first-turn，保留 manifest 字节和
   所有 SHA256。新输出目录的文件副本用于隔离缓存，不修改旧实验。
2. Static 使用反向 alpha 修正后的二值编辑区。Dilated 使用预先固定的 16 px
   方形核半径（512² 下，L∞ 膨胀），不针对 dev 调参。
3. Shuffled 在完整同 split、同尺寸 mask 池内做 seed 固定的无自配排列，不跨
   train/dev，不因推理 limit 改 donor；记录 donor ID 和哈希。不同 donor 的内容
   仍可能相同，需如实统计，不能声称每例空间提示必定改变。
4. 控制图为 RGB、三通道相同且仅 0/255；内存单通道为 [0,1]。source/mask 同坐标系，
   mask 只允许最近邻缩放。初步研究固定 512×512，拒绝隐式缩放、裁剪或错位。
5. 训练 `control_path=[reference, control_mask]`，显式 full-size、无随机空间增广，
   不使用 loss mask，不改变 flow loss/网络。推理 `image=[source, mask_RGB]`，
   不做最终 RGB 合成。No-mask 默认路径不变。
6. 条件索引记录方法、半径/seed、manifest、原 mask/donor 和生成图 SHA256。
   配置生成、worker、推理和评估分别核对；缺图、改图、错序或配置不一致必须失败。

`EditInput` 仅包含可部署输入；`ControlBundle` 是明确编译的模型条件。
`EditSupervision.target` 仅供训练/评估，推理 loader 不打开 target/oracle。
未实现的 strength/soft/role 条件明确报错，不静默忽略。

## 模块与实施顺序

- `conditioning`：二值变换、图序/数值契约。
- `datasets`：split 内 donor、条件导出/索引、哈希与推理输入加载。
- `training`：双控制配方与调用边界校验，保留设备适配和异常清理。
- `backends` / `inference`：双参考调用、输入协议与逐例条件来源记录。
- `evaluation`：条件 provenance，沿用原像素诊断与相同样本/预算比较。
- Mask 实验画廊同时显示原始标注与按记录尺寸重建的实际控制输入；No-mask 原布局保持
  不变。新视图另存目录，保留初次画廊，避免误将原始 Mask 当作 Dilated/Shuffled 输入。
- `infra` / 薄 `scripts`：可重复 smoke/初步对照，始终检查空闲 GPU 并持有租约。

先提交设计，再实现 CPU 可验证路径与测试，通过后提交；随后最小 GPU smoke，
通过后再同预算初步对照。失败保留配置、异常、显存和状态，新尝试使用新目录。

## 验收与初步对照

CPU 覆盖二值/膨胀几何、shuffle 稳定性与 split、缺图/哈希/错序拒绝、训练参数
一致性、No-mask 回归、不读取 target、后端转发顺序、记录兼容。运行 Ruff、Pytest、
安装与 CLI 检查；每个自有代码文件不超过 500 行。

GPU smoke：实时 `nvidia-smi` 确认空闲，经 `image-edit gpu -- ...` 启动。实际固定
VAE 编码/位置检查，最小多控制训练及有限 checkpoint；同 source/prompt/seed 输入
Static、零图、Shuffled，追踪双参考编码和 mask token/输出差异。敏感性只证明
条件被消费，不证明正确空间编辑或方法质量提升。

初步对照固定 150 steps、batch 4、rank/alpha 16、seed42、512²，原 AdamW/lr1e-4、
flowmatch/sigmoid、gradient checkpointing、BF16；三个方法从同一 Base 开始，不
续训 debug/No-mask adapter。固定相同 12 个 dev ID、50 steps、CFG4。完整核对数据、
训练/推理预算与版本后，复用 Stage 0 No-mask checkpoint/输出。

同更新/样本预算不等于同 FLOPs：双参考增加 token 和计算，单列真实耗时与峰值。
先测单卡，再按独立租约并行；正式比较不自动改变 batch。记录 checkpoint SHA256、
失败、逐例像素诊断和画廊；单训练 seed/小样本不产生方法有效或研究 Go 结论。

## 当前状态

第一次 CLI 配置写出遇到 StrEnum/YAML 序列化错误，训练在缺配置预检退出，未执行
GPU forward。已修复为字符串并补充写出/重读回归，失败日志保留，另用新报告重试。

联合条件导出/校验、双参考配置、实际 batch 边界、推理转发与评估 provenance 已实现。
130项完整环境测试（111.77s）、Ruff、依赖/仓库卫生及十个CLI/薄入口通过。
2-step/batch1 训练、真实 VAE/attention tracing、Static/零图/Shuffled/重复采样通过；
2-step/batch4 容量检查也完成，峰值 reserved17.09 GiB。编码对照先匹配 BF16 权重
精度，测得数值差异而非字节相等，详细值见 [实际报告](stage1_report.md)。
三组正式配置通过预算验证：与 Stage 0 No-mask 的完整配方、实际 config/checkpoint
SHA256、200/30数据和更新预算一致。三组150 steps/batch4、固定12例重载/配对评估
和两版画廊均已完成；checkpoint与真实数据训练后再次校验。150-step Static的同
checkpoint Mask消融也通过，常规/独立重载PNG一致。真实结果及局限见报告，不据
单seed/小样本像素诊断宣称性能提升，不提前进入Role或Qwen开发。
Stage 0 完整记录仍在原架构 worktree，详见 Stage 0 报告。
