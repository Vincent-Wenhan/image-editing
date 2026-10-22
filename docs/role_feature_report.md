# Independent Role Feature Conditioning：真实GPU与研究验收

2026-10-23，Asia/Shanghai。**工程可行性通过；当前残差注入/20-update预算下No-Go扩大
训练**。真实4B前反向、有限梯度、角色/空间置换、零条件及重载已通过；四方法×双
独立训练seed、96张固定Dev预测均完成。Full没有在两个seed同时满足预先设定的
T/D改善、P不恶化和身份容差门槛。此结论不等于所有独立角色特征方案被否定。
真人评分仍0条/pending，封存Test没有用于模型推理、画廊、评分或选择。

2026-10-22的GPU资源阻塞已解除；旧彩色Oracle、灰度Geometry负面结果及全部模型、
数据、checkpoint、盲评包保留。本轮没有增加数据或训练预算，没有Predictor/Qwen。

## 执行来源与协议

设计先于实现，见[预注册](role_feature_design.md)。训练、真实GPU探针与全部推理
冻结在独立worktree `image-editing-role-feature-run-02`，执行提交
`b256692e6686f30f7829cc72a969e57e11650b7b`。114个source/config文件实际SHA与
Git对象及进程快照一致。CPU评估单独冻结修复提交
`ad9d5c9415adcff26087b4e3a462dcb04cede474`，完整执行记录绑定新源码摘要。

当前资产根为role-feature worktree的 `outputs/role_feature_20261023/`，正式实验
`study_01/`；所有目录首次写入，不覆盖已运行结果。原失败和22日资源预检单独保留。
四方法constant/binary/old_new/full_role、训练seed211/223各20updates，batch4、
rank16，LoRA lr1e-4/encoder lr1e-3。每组80次样本曝光、13,312个encoder参数，
总可训练23,114,752；512²、1024 feature/1024 source/1024 noise tokens。

四组使用相同FP32编码器、单source RGB参考、位置编码与注入路径。Constant不提供
GT有效域边界；Binary提供Oracle T∪D，Old/New提供占据方向，Full增加D/P。
实际逐update的样本ID、noisy latent SHA与timestep完全相等（每个训练seed内部）；
初始化encoder SHA也相同，两个seed的初始化不同。无数据增强/隐式resize或target
RGB条件读取。没有用旧双RGB参考方法与本轮不同结构直接作公平质量排名。

Dev固定12例，4个场景×强度0/0.5/1；采样20steps、CFG4、ID采样seed42。
Dev来自前轮开发数据，不声称全新未见验证。Test只做哈希、分组与标签完整性校验。

## 真实失败、修复与资源安全

- 22日8卡有计算进程，两个正常UUID租约预检失败，未启动GPU训练；原记录仍在
  outputs/role_feature_20261022/lease_preflight_01、02。这是资源阻塞，不是OOM。
- 23日首轮沿用冻结73ba397，在extra-params初始化时报TrainConfig没有seed属性。
  0个batch/0次融合、没有checkpoint，109.39s；peak allocated14.852/reserved
  14.869GiB。原trace与失败报告留在22日study/smoke_full_role_211。异常安全释放，
  没有cleanup/telemetry错误。修复只从已校验YAML显式传seed给全局RNG与适配器，
  不改上游TrainConfig、网络或loss；新目录重试。相关13项回归23.60s通过。
- 完成96例后首次CPU分析报KeyError('identity_mae')，analysis_01的日志和部分CSV
  保留；没有测得该失败的总耗时。原fixture未连接实际指标schema，已补真实
  region_metrics→aggregate→gate回归。身份门槛读取s=0子集的预测target_mae，
  **identity_target_mae是数据source-target距离，不能衡量预测身份破坏**。
  容差仍1e-4、T/D/P标准不变。评估在新analysis_02重算，173.22s；未重训/重采样。

每个GPU工作均经实时nvidia-smi检查与正式 `image-edit gpu -- ...` UUID租约，显式
CUDA_VISIBLE_DEVICES；单卡smoke/容量通过后最多四卡并行。推理首波避开了临时
出现的GPU0进程，后续只在其退出且重新空闲时分配。未中断他人任务；全部GPU任务
正常完成并释放。无OOM、无清理或遥测失败；不启用自动batch扩大。

## 真实4B工程验收

|任务|实际更新/batch|耗时(s)|峰值allocated/reserved(GiB)|结果|
|---|---|---:|---|---|
|最小原生训练|2/1|154.23|15.022/15.037|成对checkpoint、梯度通过|
|重载GPU探针|真实前反向/采样|85.05|15.863/16.135|全部契约通过|
|容量训练|2/4|168.05|15.022/15.037|有OOM余量，允许预定batch4|

时间包含模型初始化、校验及缓存，显存是整个进程的峰值，训练加载text encoder
可能主导峰值；没有把这些数字称为纯训练step耗时或推断可用最大batch。

GPU探针实际验证：

- 零初始化与预训练Base PNG精确一致；禁用角色条件时No-mask和Static Mask路径
  精确一致；训练后的全零特征Transformer输出相等。后者不等于全P身份编辑无误差。
- 实际VAE只编码一个[1,3,512,512] source；特征不进Image VAE，noise T=0与source
  T=10各1024 tokens，按实际空间IDs对齐，角色/文本/source参考未混入噪声索引。
- old/new交换、D/P交换、空间移动分别产生Transformer输出Linf差
  0.046875/0.03125/0.03125，不能静默忽略条件；这是噪声预测差，不是质量改善。
- categories、position、hidden和output六组参数的真实梯度全部有限且非零；两步
  原生训练已让上游层获得梯度，保持原flow-matching/sigmoid objective。
- LoRA及FP32 encoder严格成对加载，再保存/重载PNG精确相等。固定探针样本
  dev_2026102201_000_s001，PNG SHA为
  b02f2cb6f386c7ebdcf0183b5bf25977a7e05db8042a72aa27e239c7dc6e2b26。

## 匹配训练与checkpoint

|训练seed|方法|20-step总耗时(s)|峰值allocated/reserved(GiB)|
|---|---|---:|---|
|211|constant|309.05|15.022/15.037|
|211|binary|311.81|15.022/15.309|
|211|old_new|300.96|15.022/15.037|
|211|full_role|300.83|15.022/15.037|
|223|constant|353.90|15.022/15.037|
|223|binary|354.14|15.022/15.309|
|223|old_new|362.23|15.022/15.309|
|223|full_role|341.65|15.022/15.309|

|训练seed/方法|LoRA SHA256|Encoder SHA256|
|---|---|---|
|211/constant|27fe0971025a5f343b413c5e1cf01514f55030d358edec5fbb397820cf6286f5|bcdbf63fc237143a7036f80b540c814a11aeed67f49d7e9796f4c97c90242c39|
|211/binary|be9eb90af3f87ff7c4795f0bdcc06908334d0efbad367aff37794fe0d2333e0f|13e67a7f5b38bd14b19c2692fa5779493ef6703628319f8aa420d9c7cc707969|
|211/old_new|ed44f9f90770093de259bb245581a15af0f7277eadeb733d49e3067adb366984|d3431641c46fbcbc508df1bb6a03392c997b9a685a4ca4f4827310cdbde8f559|
|211/full_role|bb9d96f45d057bf6265d3e0f06a9630bbde3cd61ce79613ade9d5a300585add9|611592842848f6eaa5c1a1c4003cf2aadd6c374d0ace826bc2f992ad7458471f|
|223/constant|c1783b3a32857a6fbb5a8ef0cb329458294bb5eece43d0ae11e29b976facb4d0|c000842448fff4300487350681147ff298e342a35681671c58e4b0b31d1d1930|
|223/binary|d8812a64d85aa6a7973cebe7f8ae2ec38241bf15dd4bfa4ff1c29c7be5ae2470|ea2432580b21c3fd14cac3316a5b481ee2c3074e8b053ebf3f54ae4c7b2f653b|
|223/old_new|e9a654270f2927ff7070fa643b23f09bbb40bdd220482f2afcb1b8c11b1750ca|dec5feed23f349c32b629427f95ee488fa0155c62116f814b5022564251ff2e8|
|223/full_role|5b2b6eee4f9233474ad20f0fbee69deab32a5ea26d8aa2f47f2a37953e216c92|6276dff5ecfd1ffe1989385afbc599f15af7e68bc54851267032f44d47cf3fbf|

每组最终路径为 `study_01/train_<method>_<seed>/lora/train_<method>_<seed>/`，
文件名 `train_<method>_<seed>.safetensors` 及同名前缀`.features.safetensors`。
原始train_report.json记录实际config SHA、160个有限LoRA张量、完整梯度和每步轨迹；
matched_training_01.json重新核对真实权重字节及预算，不仅比较YAML文字。

## 固定Dev重载推理

|训练seed/方法|进程耗时(s)|12例采样耗时合计(s)|
|---|---:|---:|
|constant_211|261.05|130.76|
|binary_211|262.67|132.17|
|old_new_211|262.13|131.46|
|full_role_211|262.24|131.48|
|constant_223|247.64|129.94|
|binary_223|247.61|130.42|
|old_new_223|248.37|132.34|
|full_role_223|246.76|130.95|

所有8组各12张、共96张512² RGB预测；峰值allocated15.594/reserved16.262GiB。
进程耗时包括校验与加载，采样合计只汇总run中每例记录；两者不能混为纯模型吞吐。
固定输入的四方法输出都不是完全相同的PNG，Full对其余方法全图MAE均值约
0.000674–0.001042，但视觉结构很接近。运行时每例都记录了非零残差和至少20次
真实融合；不同方法PNG有差不能替代同checkpoint置换测试，后者由GPU探针验证。

## 像素结果与预设门槛

以下图像值归一化到[0,1]。非零强度8例先在同场景内平均，再四场景等权；身份列
只取s=0的4例预测target MAE，此时target=source。T/D分别对target、P对source，
不把背景保留当编辑完成，也不将source不动的误差基线当预测身份质量。

|训练seed|方法|T target MAE↓|D target MAE↓|P source drift↓|s=0预测MAE↓|
|---|---|---:|---:|---:|---:|
|211|constant|0.083134|0.266279|0.019464|0.022340|
|211|binary|0.083180|0.266456|0.019279|0.022282|
|211|old_new|0.083369|0.266172|0.019543|0.022538|
|211|full_role|0.083218|0.265628|0.019441|0.022798|
|223|constant|0.085740|0.269247|0.016167|0.018757|
|223|binary|0.085871|0.269753|0.016058|0.018335|
|223|old_new|0.085769|0.269570|0.016324|0.018707|
|223|full_role|0.085822|0.269701|0.016218|0.018667|

|训练seed|Full减去对照|ΔT [95% CI]|ΔD [95% CI]|ΔP [95% CI]|
|---|---|---|---|---|
|211|constant|+0.000083 [-0.000676, +0.000900]|-0.000651 [-0.001576, +0.000072]|-0.000023 [-0.000379, +0.000298]|
|211|binary|+0.000038 [-0.000696, +0.000542]|-0.000828 [-0.002184, +0.000142]|+0.000162 [-0.000478, +0.000635]|
|211|old_new|-0.000151 [-0.000642, +0.000227]|-0.000544 [-0.001793, +0.000222]|-0.000102 [-0.000378, +0.000174]|
|223|constant|+0.000082 [-0.000210, +0.000424]|+0.000453 [-0.000067, +0.000973]|+0.000051 [-0.000078, +0.000180]|
|223|binary|-0.000049 [-0.000613, +0.000397]|-0.000052 [-0.000327, +0.000266]|+0.000160 [+0.000108, +0.000204]|
|223|old_new|+0.000053 [-0.000163, +0.000268]|+0.000131 [-0.000183, +0.000407]|-0.000106 [-0.000351, +0.000034]|

|训练seed|方法|旧独占target MAE↓|新独占target MAE↓|
|---|---|---:|---:|
|211|constant|0.044799|0.123385|
|211|binary|0.045057|0.123243|
|211|old_new|0.045402|0.123241|
|211|full_role|0.045479|0.122815|
|223|constant|0.045020|0.129045|
|223|binary|0.044552|0.129901|
|223|old_new|0.044976|0.129224|
|223|full_role|0.044758|0.129514|

区间为Full减去对照，负值更好；bootstrap2000次/seed42，仅反映**固定训练seed内
四个场景**的不确定性，不是训练seed方差或总体显著性证明。多数区间跨0；223的
P相对Binary为正，但场景覆盖仍小，不能作广泛性能断言。没有用区间事后放宽门槛。

seed211：Full的T不优于Constant/Binary，P不优于Binary，身份对三个对照都超出
1e-4容差；虽D略好，整体门槛失败。seed223：T/D不优于Constant/Old-New，P不优于
Constant/Binary，身份相对Binary也超容差；整体失败。两seed都没有完整通过，
因此No-Go当前注入/预算的扩量。小差异与区间不能证明所有角色语义无效。

旧/新独占target MAE是位置区域代理，source距离另存8张geometry CSV及summary；
它们不识别物体，不给出重复率。Full在211新位略好、旧位略差，223也没有一致优势。

## 非盲诊断与真人准备

inspection_02含8张完整scene/seed联系图及全部96张预测的实际SHA；索引和
agent_observations.json记录代理非盲观察，**不是独立真人评分或成功率**。

- 两seed的000_s001仍近源位置；000_s002出现额外竖直偏移。
- 001_s002的圆柱和阴影放大；002的强编辑发生物体形变/位置偏移，未保持纯平移。
- seed211的003_s001出现脱离物体的额外黑影；两seed的003_s002被背景棋盘带截断。
  额外阴影不能记作重复物体，本轮联系图没有目视确认清晰的双物体主体。
- 身份场景布局大体保留，未目视复现旧彩色Oracle的大面积色板；仍有约0.019–0.023
  的全图身份MAE。无RGB条件图不保证所有外观复制或身份问题已经解决。
- 四方法错误形态接近，未观察到Full稳定的语义优势。实际真人类别与严重度仍未知。

human_feature_01/seed211、seed223各保留独立bundle、三套空白panels/CSV，每包
48任务/人，共96任务/人；任务类型注释在tasks.csv。真实评分0条，三位完整独立
评审均pending，没有导入或自动填写分数、人工置信区间。仅分享panels，私有映射
与代理诊断不提供给评审者，详见[使用指南](human_review_guide.md)。所有旧人工包保留。

## 检查、数据与版本审计

本机最终 **232 passed / 149.59s**，无跳过；Ruff check/format通过，43项原/新CLI
检查和修复后的评估CLI帮助通过，pip check无缺失依赖。新回归连接真实指标协议，
能识别source=target但预测破坏身份的情况；不是把手写假字段当实际接口。
之前231项/147.34s和22日230项/272.27s日志保留，不将它们混报为最终232项。
小尺寸随机初始化的真实Diffusers FLUX.2 CPU模块检查亦保留，不充当4B权重实验。
[评估修复CI](https://github.com/Vincent-Wenhan/image-editing/actions/runs/39834629455)
通过216项/11.64s，16项因CI未装Torch明确跳过；本机232项全部执行。PR与push
两次CI均成功（39834619043、39834629455），原始日志保留ci_evaluation_fix.log。

新resources_audit.json实际重算23个现有模型/配置文件SHA，与前轮已验证固定
revision摘要对照，37.40s，无重新下载。第三方tracked文件均干净；版本如下：

|来源|固定revision|
|---|---|
|FLUX.2 Klein Base 4B|a3b4f4849157f664bdbc776fd7453c2783562f4d|
|ai-toolkit|a055947d56a156411abb4ef0301b35d0fe5f51a5|
|Inter-Edit|220c1dc26d8401cae160a6b25e48198f85bdd115|
|Diffusers源码|d791c5c024014784df4a8dac2601e19fb4d300fc|

Python3.11.17、Torch2.6.0+cu118、Diffusers0.37.0、Transformers4.57.3、PEFT0.18.1；
复用conda image-edit，各执行worktree有独立editable .venv。完整环境版本见原始记录。
数据复用role-failure worktree的scenes512_01：scene seed2026102201，12/4/4独立
train/Dev/Test组，36/12/12图像对。实际哈希、反事实角色重算、valid和split隔离通过。

|资产|实际SHA256|
|---|---|
|dataset.json|1c1b2ddc7f5d1df5d9cd0bf11eee2c405d49a128f8745682737623ddf9631c21|
|train manifest|690d3c864409c5d563c69652ef3e21946e119fec60d467ddc6d1c20862e2f8f9|
|Dev manifest|b6778f19688023a6720041fe9f1a8b69fcd69e4b2d505b32784ee3a020efb29f|
|封存Test manifest|3a6f26bc9bb4896066046a5597d5fa7bea59c94ecfdbadc3f5decda1c6c8d68b|

## 可核验证据索引

下列路径均在本轮outputs/role_feature_20261023；只提交文字报告与源码，不上传资产。

|记录|内容|
|---|---|
|resources_audit.json|现有模型/依赖/第三方/data实际SHA|
|training_source_audit_01.json、matched_training_01.json|10个成功任务/8组公平性、成对权重与源码|
|study_01/gpu_smoke/report.json及gpu_smoke_execution/|真实4B前反向、有限梯度、置换、零条件、重载|
|study_01/*/train_report.json、training_execution/|时间/显存/checkpoint/完整样本噪声轨迹|
|inference_source_audit_01.json、study_01/*/dev_execution/|96张Dev PNG实际SHA/运行时融合/冻结源码|
|analysis_01/、analysis_01.log|原CPU字段失败，部分产物保留|
|analysis_02/summary.json、analysis_execution_02/|修复后完整指标、配对区间与评估源码/时间|
|inspection_02/|八张非盲联系图、逐文件SHA和具体案例|
|human_feature_01/|两seed三人空白包/私有映射/任务注释，真人pending|
|pytest_final_02.log、ruff_final_*、cli_checks.json|232项本机回归、静态与43项CLI原始结果|

## 研究决策与剩余风险

**工程Go，研究No-Go扩大当前版本训练**。独立于Image VAE的特征已能稳定训练、
消费、保存并重载；禁用兼容、零残差及角色/空间干预有真实证据。但在匹配容量/
token/轨迹/预算下，没有得到Full相对无信息、区域并集及Old/New的稳定额外收益。
不能用旧RGB失败反推新方案成功，也不能用一次GPU smoke代替质量结论。

不增加数据量/步数，不进入Dynamic Role Predictor或Qwen。下一步优先完成真实
独立盲评；若继续分析，只考虑利用已有checkpoint定位残差信号相对噪声、BF16
分辨性、不同时间步影响及角色方向混合，先提出新的有界假设与验收条件。
20updates低预算、单次空间输入残差、token池化混合、BFL训练与Diffusers采样的
数值差别、四场景/两seed覆盖及重复使用Dev仍限制结论。全零条件等价已通过，
全P身份保持未达到继续门槛，两者不能混淆；封存Test继续不用作选择。

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
