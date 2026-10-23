# T/D/P Role Conditioning：可学习性与失效机制

2026-10-23，Asia/Shanghai。原注入方案的四方法双seed训练、356图及实际SHA评估已完成，
**原方案在本预算下No-Go**：两个seed都未通过少样本fit、角色利用或D增量门槛。
唯一候选channel_rms已预注册、实现并通过CPU及全部GPU门禁；正式对照运行中，收益尚待验收。
不能把loss下降、非零响应或工具成功当作语义成功。全部历史资产与失败保留，Test封存，真人pending。

## 来源与范围

PR #6接受头ac2f8e7经252 tests/218.67s、55 CLI、96图指标/配对区间重算、资产及
空白盲评复核后正常merge为main 8b261bb7abbcdeaa23230f6686ec472d3d6fb909，
14:39:01；原CI39907007950/39906853717 success。没有force push。
审计保存在role-feature-diagnostics worktree的outputs/pr6_merge_audit_20261023。
新分支feat/role-learnability从该main创建。本阶段不训练Predictor、不迁移Qwen。

新资产根为image-editing-role-learnability worktree的
outputs/role_learnability_20261023/；数据、模型、图像与权重不上传Git。
基线训练冻结2f439d6、推理aef9662、指标2375cd0，均独立worktree执行；
channel_rms运行冻结2375cd0。设计先行见role_learnability_design.md与
role_learnability_repair_design.md，实际源码逐文件SHA随每次execution/报告保留。

复用原512² Blender场景seed2026102201及cohort：12/4/4组、36/12/12对Train/Dev/Test。
按Train group_id排序取前两组，实际scene005/009，各0/0.5/1，共6例；不是按效果选例。
每组训练新建目标/source/caption视图，隔离缓存，不改canonical资产。
Dev全部12例用于诊断，已在历史研究使用；Test仅完整性验证，无生成/画廊/选型。
实际manifest SHA256：

|split|SHA256|
|---|---|
|Train|690d3c864409c5d563c69652ef3e21946e119fec60d467ddc6d1c20862e2f8f9|
|Dev|b6778f19688023a6720041fe9f1a8b69fcd69e4b2d505b32784ee3a020efb29f|
|Test封存|3a6f26bc9bb4896066046a5597d5fa7bea59c94ecfdbadc3f5decda1c6c8d68b|

固定Base revision a3b4f4849157f664bdbc776fd7453c2783562f4d，ai-toolkit
 a055947d56a156411abb4ef0301b35d0fe5f51a5，Inter-Edit
220c1dc26d8401cae160a6b25e48198f85bdd115，Diffusers d791c5c024014784df4a8dac2601e19fb4d300fc。
23模型文件实际SHA重算28.51s通过，外部tracked源码无修改；没有重新下载模型。
Torch2.6.0+cu118、Diffusers0.37.0、Transformers4.57.3、PEFT0.18.1。

## 公平性、监督边界与门槛

Constant/Binary/Old-New/Full，seed307/311，各从相同Base独立初始化，120updates、
batch6，正式共960updates；每update实际恰好覆盖6个Train ID各一次。
LoRA rank/alpha16、lr1e-4、encoder lr1e-3；13,312 encoder参数、23,114,752总训练参数，
feature/source/noise各1024 tokens。原flowmatch/sigmoid及训练目标不改变。
两seed内实际初始参数、每步ID/噪声latent/timestep、容量和优化配置逐项匹配。
每组保存update0/20/40/60/80/100/120的LoRA+encoder，逐对验证实际SHA，未自动删旧快照。

所有Train/Dev使用同一中性指令，唯一文本与数值预算在role_learnability.yaml。
同scene不同强度共享source、prompt与初始噪声，seed按group_id计算；Constant及
Full update0的同组三种强度PNG逐字节相同。原/新指令结果不混作同预算比较。
生成只使用source和显式Oracle字段old/new/D/P/valid；target只进入训练监督及独立
teacher/梯度评价，不是reference、文本或Role Encoder输入。Oracle不是部署条件。

Full初始/20/60/120学习曲线、最终四组原条件及Train干预，合计356图；Full固定
Train/Dev各首个非零例做三档真实CFG状态回放、七层响应和三档监督梯度诊断。
角色交换、空间置乱和zero均共享相同输入；置乱保持pooled-valid和值集合。

预定fit：T/D对source参考各降低25%，T≤0.05、D≤0.10、P≤0.01、identity≤0.01，
T/D同强度contrast MAE≤0.75。角色利用还要求Full对Constant和自己的zero/shuffle
的T/D各改善5%、P/identity增量≤0.001；D独立价值要求对Old/New再改善5%。
contrast比较(pred_s-pred_0)与(target_s-source)，完美0/有向响应1，Constant1/0。
Train仅2场景；Dev四场景bootstrap2000/seed42，不代表训练seed总体不确定性。

## 原注入：真实结果

非零Train4例按scene等权；T/D/P与identity为MAE、越低越好，contrast完美为0。

|seed|方法|T|D|P|identity|T contrast|D contrast|
|---|---|---:|---:|---:|---:|---:|---:|
|307|Constant|0.164609|0.330334|0.011981|0.013514|1.000000|1.000000|
|307|Binary|0.166932|0.310654|0.011046|0.012833|0.998833|1.000742|
|307|Old/New|0.166175|0.310536|0.012727|0.013303|0.992249|0.998059|
|307|Full|0.168249|0.311111|0.009029|0.010652|0.999039|1.002119|
|311|Constant|0.168569|0.336256|0.009932|0.011176|1.000000|1.000000|
|311|Binary|0.168110|0.331341|0.011263|0.012878|1.001204|0.999403|
|311|Old/New|0.169576|0.335341|0.009984|0.011102|0.999384|1.000023|
|311|Full|0.169695|0.334928|0.010695|0.012247|1.001761|1.000240|

不编辑source的非零Train T/D为0.168721/0.333698。Full在307的T仅改善约0.28%，
311的T还略恶化；两seed的contrast均接近1，几乎没有学会随角色改变编辑强度。
307相对zero有D收益，但对shuffle没有，且T失败；311也没有跨全部干预的T/D收益。
Full未显示超过Old/New的D增量。不能只挑一个D均值或P保留宣称角色有效。

Full从0/20/60/120的Train T/D：307为0.170174/0.301652、0.170742/0.287917、
0.169707/0.298439、0.168249/0.311111；311最终0.169695/0.334928。
T contrast始终约1；增加到120updates未形成正确条件方向。
Full first20→last20训练loss：307为0.036867→0.024777，311为0.034554→0.029308；
Constant分别0.036864→0.024762、0.034555→0.029141，下降相近。

监督teacher在t=0.5的Train T/D velocity MSE从初始0.107981/0.120505降为
307的0.069384/0.082246和311的0.070595/0.079891。它是含noisy target状态的
未加权监督诊断，不是配置中的原加权loss，也不证明生成依赖角色或到达目标位置。

Dev非零8例、4场景等权，Full的307 T/D/P/identity为
0.113320/0.329625/0.013579/0.015690，311为0.104303/0.239205/0.015056/0.019536。
Full减Constant的Dev T差及95%场景区间：307 -0.005151[-0.015842,0.000280]，
311 -0.002851[-0.008983,0.000511]；D差分别+0.023802[-0.001309,0.048913]和
-0.024551[-0.050166,0.001063]。D方向不跨seed稳定，区间也未排除0；
311的P漂移增量0.002153[0.000409,0.004325]。完整四方法/分强度/阴影与位置结果在JSON。

## 机制证据与局限

实际训练中梯度到达编码器和所观测的projection、双流首/中/末、单流首/中/末层。
update1类别梯度为零符合输出零初始化，后续非零；不能据首步零梯度误判断连。
307末步Old/New/D类别参数梯度RMS约5.67e-7/4.98e-7/3.38e-7，P/valid约1.63e-5。
类别/位置输入RMS比0.0773；这支持稀疏角色信号弱，不是可加的梯度贡献比例。
不能把梯度小等同于参数没学到更新：原307 Old/New/D/P类别权重变化RMS为
0.021205/0.015546/0.012312/0.012044，311为0.018431/0.011300/0.006208/0.009619。
AdamW会自适应梯度尺度；valid还有独立的输出门控路径，输入梯度不可直接当作
同量纲的语义重要性排名。encoder_updates_01保存实际FP32权重变化与文件SHA。

Full首个Train诊断例、正CFG分支：

|seed|step/t|残差/原特征RMS|Old/New差分被BF16加法丢失|交换后CFG速度相对差|
|---|---|---:|---:|---:|
|307|0/1.0|0.815%|84.06%|1.75%|
|307|9/0.890625|0.907%|83.02%|3.33%|
|307|19/0.263671875|1.253%|79.77%|7.81%|
|311|0/1.0|0.803%|86.00%|1.69%|
|311|9/0.890625|0.893%|85.05%|3.28%|
|311|19/0.263671875|1.222%|82.82%|7.25%|

回放必须逐元素复现实际轨迹，zero/disabled精确一致，source/text/noise不变。
fixed_state_audit_01还核对了跨update/方法/seed的实际6组source/text/noise及54组
teacher输入摘要一致，避免将预处理或监督噪声变化误当学习曲线。
角色交换经过七个观测层并影响输出，排除“完全未加载/完全断梯度/静默忽略”；
非零响应本身仍不是正确的空间语义。Dev首个非零例趋势相近，不能推广成全部场景统计。
固定监督t=0.1/0.5/0.9的区域梯度使用共同全图分母，保留BF16反传求和误差；
区域范数比不可相加。P面积大，稀疏T/D的驱动力较小；latent区域按16像素面积池化，
是velocity代理，不是精确像素物理归因。原训练层观测的时间与update也存在混合因素。
交换/置乱/zero可能偏离训练分布，单独不能证明角色未用；原条件的Train拟合也失败，
因此当前No-Go不依赖这些干预的单一解释。

时间暴露复核（timestep_audit_01）：原trace以0–1000计，先按Native实际BF16
cast/除1000重建模型时间，并逐项匹配已观测的训练时间及采样状态。720次样本曝光中，
307的t≥0.8/0.9/0.95分别46/5/0次，311为75/18/1次；20-step推理对应13/9/6步。
高噪声条件学习覆盖少，是与弱信号并列的待检验因素；推理节点密度不必等于训练
分布，尚无时间采样对照，不能将这一差异判为训练器bug或唯一失效原因。

非盲检查baseline_galleries_01的四例：Train005_s001四方法均停留旧位置；
Train009_s002从0到120仍未到target，阴影形状变化；Dev003_s002在zero、交换和
shuffle下仍停在旧位置；Dev000_s000本应identity，311 Full仍移动/改变物体与阴影。
这些是agent观察，未写成人工评分。Train定位4/4非零例可用，Dev仅5/8；其余弃权。
当前模板未提示重复候选，但不是可靠物体计数或“没有重复”的证明。
画廊彩色十字是评价叠图，不是生成的角色颜色。68张联系图和所有原图链接保留。

## 唯一候选与当前执行状态

已预注册channel_rms：现有类别线性层前逐样本逐通道空间RMS归一化，统计stop-gradient，
空通道分母1；原位置编码、网络/容量、tanh上界和valid门控不变。默认none/旧权重
保持兼容，constant全1严格不变。修复只用Train观测确定，不依赖封存Test。

新增20项CPU回归35.55s及5项归一化/重载/收益门槛测试20.49s通过。
正常租约的两步batch1训练245.45s、peak allocated15.022/reserved15.037GiB；
真实前反向、置换、zero、空间与重载检查155.02s、reserved16.268GiB通过。
batch6容量329.06s/reserved18.355GiB、新增梯度诊断176.71s/reserved16.281GiB通过，
正式四组双seed按pipeline_01.json顺序运行，尚无候选性能结论。
计划仅追加同四方法双seed各120updates，共960；正式总上限1920，不延长或搜索第二改法。
对照仍需原fit/角色利用/D增量门槛及两seed相对原Full T/D各改善5%、P/identity增量≤0.001。

## 工程、失败与真人状态

272 passed/205.96s，本地有Torch；Ruff check/format及65 CLI通过，CI39929339061 success。
原真实GPU：训练smoke198.78s、前反向/重载105.33s、
batch6容量266.36s，peak reserved15.037/16.268/18.047GiB；新诊断171.49s/16.281GiB。
基线八个正式训练各1238.65–1271.56s，peak allocated15.069/reserved18.047GiB。
时间含加载/I/O，不作为算法吞吐比较。GPU前实时SMI，正常UUID租约跳过占用卡，
显式CUDA_VISIBLE_DEVICES；未中断他人任务，没有OOM或旧资产覆盖。

study_01在AcceleratedOptimizer step-hook边界AttributeError，0更新；study_02完成
2updates/211.53s后保存失败，因上游post_save_hook给出optimizer.pt导致错误encoder绑定。
两次失败中的该绑定无效，不用于模型比较；LoRA/错误encoder/trace完整保留。
自有代码改为内部optimizer注册和真实network.save_weights边界，不改第三方。
asset_audit_01读取editable元数据vcs_info失败；02改为核对实际本地Git，23文件审计通过。

human_audit_01核验既有6份独立顺序表、各48任务，共288行仍空白；真人0位、0条。
工具的导入/合并/配对统计/CI经过单元回归，真实导入及≥3位独立真人结果仍pending。
这些包是上一阶段冻结干预研究，不充当本轮新指令实验的真人证据；不自动邀请或代填。

Full最终LoRA/encoder SHA：

|seed|LoRA|encoder|
|---|---|---|
|307|4177d77c0f9270e901b142d7a4236b3ef5d30eee06bf68dde8bd912cd9cc56b6|23c744e8c88cbb1daa7ee34ab8f1ea0cb304a011bbf8f8404863cf582cb5ab9d|
|311|dbadde1ab2f92a1d1bdd4600ed9331d54ef4c12ac6e895647af8fa5927827927|d6ed3cd971ca379a57a57ec74fa396852ef02498a87981173cd0b732246760f3|

所有方法权重在study_03/train_<method>_<seed>/lora/<job>/；train_report.json保存完整
checkpoint路径、每步loss、七组成对快照、显存/时间/失败和实际SHA。主要资产：
plan/protocol与*_phase.json；baseline_analysis_01/summary.json、training_loss.csv及
五份分组summary；baseline_galleries_01/；human_audit_01.json、asset_audit_02.json、
pytest_full_02.log；normalization_study_01/及其pipeline/execution，后续结果另存新目录。

当前结论仅针对原注入/本预算：没有展示可靠的Old/New/D/P条件编辑；有限梯度和层响应
证明工程连接有效，尚未证明语义利用。弱差分/BF16和区域梯度不平衡有证据，训练不足、
注入位置和目标的影响仍未完全分离。T/D/P作为反事实数据与评价值得保留；
是否值得继续该条件方法，待唯一有界候选完成后按预定门槛收尾，不扩大训练。

补充数值复核：Constant两轮初始LoRA/encoder张量完全相同，seed307四组全部120步
的ID/noisy latent/timestep、初始化和容量逐项一致。相同已训练权重、6×1024个
Constant字段下，真实GPU前向及全部参数/输入梯度逐元素相同（constant_equivalence_02，
0.708s，轻量编码器检查）。独立重复训练的Constant在update20仍有最大0.0009765625
的LoRA保存值差，未逐核定位来源，不把它当归一化收益或声称训练逐位可复现。
原/新八组的实际初始化/完整轨迹/容量检查均已通过paired_train_all_01，正式共1920updates。
专项检查第一次指定卡的租约预检拒绝、0 GPU工作；旧日志保留，第二次仅使用实时空闲卡。
