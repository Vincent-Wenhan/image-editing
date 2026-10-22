# Independent Role Feature Conditioning：验收与待运行项

2026-10-22，Asia/Shanghai。当前结论：**Hold；No-Go 扩大训练**。独立特征接口与
CPU检查已经通过，但服务器8卡被其他计算进程占用，正常UUID租约预检失败。
尚无本轮4B GPU前向/反向或编辑质量证据，不能判定方法有效，也不能宣称该方法
已被实验否定。彩色Oracle和灰度Geometry的既有负面结果继续保留。

## PR #4与main复核

已复核role-failure分支最终提交c8f2b52，最新PR/push CI均成功（39743041254、
39743033856），合并前MERGEABLE/CLEAN，无冲突。重新执行214 tests/194.29s、
Ruff、26 CLI，并实际重算旧固定干预/双seed pilot指标，核对checkpoint/config
SHA、数据分组/反事实/valid及两个真人pending包，支持原报告的受限No-Go结论。

未误提交数据、图片、模型、checkpoint或第三方源码；合并前资产与源码审计通过。
证据保留在role-failure worktree `outputs/pr4_merge_audit_20261022/`，含evidence、
实际重算结果、原始测试日志和merge receipt。没有覆盖原实验目录。

[PR #4](https://github.com/Vincent-Wenhan/image-editing/pull/4)正常merge为
`b73492f55ab24b6f3e5c3da0ce4aa24b8cac7b13`，时间16:40:40+08:00，无force push。
合并后[main CI](https://github.com/Vincent-Wenhan/image-editing/actions/runs/39749531815)
也成功。新分支feat/role-feature-conditioning从该main创建，在独立worktree开发。

## 本轮已实现

设计先于代码记录在[预注册](role_feature_design.md)。复用固定FLUX.2 Klein Base 4B、
ai-toolkit与数据管线，不改external、flowmatch/sigmoid目标和现有No-mask/Mask入口。

- 独立的old/new occupancy、D、P、valid五通道单位图，有限浮点、同像素几何，
  语义值不超过valid；Full要求D+P≤valid，当前硬标签D/P互斥；old/new允许重叠，
  身份样本仍保留真实占据。
- 每16像素面积池化、五类别嵌入及二维Fourier位置编码，64维FP32 MLP，共13,312
  个可训练参数；128维输出零初始化、tanh限幅0.25，valid门控使全零条件始终为零。
- 按实际(T,H,W,L) IDs对齐，仅向噪声token加残差；source/文本与位置ID不变。
  角色特征不进入Image VAE。BFL与Diffusers分别适配，不改模型内部模块或loss。
- 原trainer extra-params hook将FP32编码器加入同一optimizer；保留原异常、失败
  报告与GPU统计，关闭hook，禁止漏编码器的中断续训。只支持fresh单卡实验。
- 编码器独立safetensors，绑定LoRA实际SHA、base revision、mode、schema和配置；
  缺失或错配拒绝。显式oracle接口与普通EditInput/推理分离，不读取target RGB。
- Dev-only推理记录兼容原run格式并增加特征元数据；评估实际重算encoder/LoRA/
  feature SHA。四组审计实际样本ID、noisy latent SHA、timestep轨迹及容量/token。
- 分列T/D/P、身份和旧位残留/新位误差代理，按scene等权及配对bootstrap；没有
  将像素代理当物体计数或真人成功率。空分区保持undefined，不能用于通过门槛。

## 已运行的CPU检查

完整本机回归 **230 passed / 272.27s**，Ruff check/format通过，pip check无缺失依赖。
安装后的包可在checkout外导入，全部模块和CLI帮助不初始化Torch/CUDA。
完整日志在新worktree `outputs/role_feature_20261022/pytest_final.log`；CLI记录
在cli_checks.json，依赖检查在pip_check.log。

测试覆盖原全部功能、新字段/哈希/防target读取、矩形网格面积池化、真实位置ID
gather与source不变、零初始化及训练后全零、两步有限梯度、角色/空间置换、成对
重载、配置/封存Test拒绝、预算门槛及失败清理保留原异常。

另实际运行**小尺寸随机初始化的Diffusers FLUX.2 Transformer CPU前反向**，
含gradient checkpointing、有限非零encoder梯度、条件置换、零条件、权重重载。
使用真实模块实现但缩小hidden/layers，不是4B权重或GPU实验，也不是模型质量证据。
核对BFL的(c,patch_y,patch_x)与Diffusers patchify/pack顺序一致。

## 数据、版本和历史资产

复用role-failure worktree `outputs/role_failure_20261022/scenes512_01`：scene seed
2026102201，512²，12/4/4 train/Dev/Test组、36/12/12图像对，强度0/0.5/1。
完整验证包含真实图像/角色/valid/evidence哈希、标签重算、背景补全有效域和split
隔离；没有新渲染、移动或删除旧数据。Dev已被前轮使用，明确属于开发数据。
Test只做完整性校验，未用于特征推理、画廊、评分或模型选择。

|资产|实际SHA256|
|---|---|
|dataset.json|1c1b2ddc7f5d1df5d9cd0bf11eee2c405d49a128f8745682737623ddf9631c21|
|train manifest|690d3c864409c5d563c69652ef3e21946e119fec60d467ddc6d1c20862e2f8f9|
|Dev manifest|b6778f19688023a6720041fe9f1a8b69fcd69e4b2d505b32784ee3a020efb29f|
|封存Test manifest|3a6f26bc9bb4896066046a5597d5fa7bea59c94ecfdbadc3f5decda1c6c8d68b|

新resources_audit.json实际核对21个Base模型文件及training VAE，与缓存的固定
revision Hub内容摘要一致，包括所有现有权重；耗时78.53s，没有重新下载。
派生text encoder仍指向已验证的原文件。第三方tracked文件均干净：

|来源|固定revision|
|---|---|
|FLUX.2 Klein Base 4B|a3b4f4849157f664bdbc776fd7453c2783562f4d|
|ai-toolkit|a055947d56a156411abb4ef0301b35d0fe5f51a5|
|Inter-Edit|220c1dc26d8401cae160a6b25e48198f85bdd115|
|Diffusers源码|d791c5c024014784df4a8dac2601e19fb4d300fc|

运行环境复用conda image-edit（Python3.11.17、Torch2.6.0+cu118、Diffusers0.37.0），
新worktree使用独立editable .venv；完整版本在resources_audit.json。原数据、模型、
checkpoint、负面实验与两个人工包均保留，真实评分仍0条/pending。没有用测试fixture
或代理观察填任何人工评价。真人操作见[指南](human_review_guide.md)。

## 准备完成但没有运行的GPU实验

新目录 `outputs/role_feature_20261022/study` 已生成独立缓存数据副本、训练配置和
不可覆盖计划：四方法×seed211/223，每组20updates/batch4/rank16，LoRA lr1e-4、
encoder lr1e-3；所有方法相同13,312 encoder参数与1024特征tokens，预期单source
参考和噪声各1024 tokens。正式相等性仍须由实际训练轨迹/运行时计数核验。

|方法|输入信息|本轮checkpoint/实际训练耗时/峰值显存|Dev质量结果|
|---|---|---|---|
|Constant Feature|全1，含位置，无GT有效域边界|pending / 未启动|pending|
|Binary Feature|T∪D并集复制四份，加valid|pending / 未启动|pending|
|Old/New Feature|旧、新、valid、valid、valid|pending / 未启动|pending|
|Full Role Feature|旧、新、D、P、valid|pending / 未启动|pending|

17:13起多次实时nvidia-smi均显示8卡计算进程，显存约41,848–43,804MiB/卡；低瞬时
util并不等于空闲。17:22:13通过正式 `image-edit gpu -- ...` 做UUID租约预检，返回：
`No idle, unlocked GPU available; no task was started`。原始stdout/traceback与执行
provenance保留lease_preflight_01/，实时快照保留gpu_availability_*.json。
没有中断他人任务、没有在占用卡上加载模型；这是**资源预检失败，不是训练失败或OOM**。

因此2-step/batch1原生训练、4B真实前反向/有限梯度/置换/空间/零条件/成对PNG重载、
batch4容量、8组20-step及96张Dev评估**全部pending**。没有新的GPU耗时、显存、
checkpoint或编辑效果可报告，不能引用旧RGB/灰度结果作为特征方案验证。

## 继续条件与风险

代码已提供门禁入口（README），不留后台无人监督训练。先获得实时空闲卡与UUID
租约，在冻结执行worktree完成2-step及feature_smoke；成功后才测batch4并最多四卡
并行。按实际SHA、相同参数/token/更新/样本轨迹审计，才允许比较两seed固定Dev。
旧位保留/新位误差代理与预选案例分列，最后由真实独立评审判断重复物体与编辑成功。

主要剩余风险：4B BF16残差可辨性、固定BFL训练与Diffusers采样的真实数值一致性、
额外optimizer参数的实际更新与保存、不同角色信息在低分辨率token内的混合、身份
保持、已有Dev重复选择及4个Dev场景的统计覆盖不足。独立于图像VAE不保证模型
不会再复制外观，也不自动解决语义或旧新位置歧义。

当前建议：**不扩大数据/步数，暂不进入Role Predictor或Qwen；允许在GPU空闲后
继续已预注册的最小可行性验证**。只有Full在两个seed通过T/D/P/身份和Old/New额外
收益门槛、没有明显位置/重复退化，才能考虑provisional Go；真人未完成前仍不扩大。
