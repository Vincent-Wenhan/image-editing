# Role-aware Conditioning V2：独立特征最小可行性

2026-10-22（Asia/Shanghai）。起点main `b73492f`，PR #4已正常合并。合并前最新CI
通过，重新执行214项测试/194.29s、26个CLI、实际数据/预测/checkpoint与No-Go复核；
新记录在role-failure worktree的outputs/pr4_merge_audit_20261022。旧彩色/灰度负面
结果及所有模型、数据、盲评包保留。本文先记录设计，实际运行另写报告。

## 范围与数据

复用已经验证的512数据：scene seed2026102201，12 train/4 Dev/4封存Test组，
强度0/0.5/1，共36/12/12对。使用同一cohort seal及真实SHA。Dev已经用于前轮研究，
这里明确用于开发，不声称新未见验证。Test只做完整性校验，不生成模型输出、不做
画廊/人工评分/超参选择。不生成新数据或扩大步数，不训练Predictor，不迁移Qwen。

## 输入与编码

新增显式上界输入，含old occupancy、new occupancy、D、P、valid五个同尺寸
浮点通道，有限[0,1]，valid外语义通道为0。old/new允许重叠，s=0时仍表示真实物体
位置，P表示无需编辑；不强制五通道和为1。原EditInput只含source/instruction，
训练target继续独立加载。条件读取只选NPZ占据字段和roles/valid，禁止打开target RGB。

每16×16像素面积池化成一个特征token；512时网格32×32/N=1024。四组共用同一
FP32编码器：五个类别嵌入、二维坐标Fourier编码、64维MLP、128维零初始化输出层。
valid单独门控，无效token输出0；全零输入在训练后也严格输出0。输出tanh限幅0.25。

|对照|五通道内容|信息边界|
|---|---|---|
|Constant Feature|所有通道固定1，不读入valid边界作为特征|只有常量/坐标，无角色信息|
|Binary Feature|T∪D的同一bit复制四份，最后为valid|Oracle区域并集，无旧新方向|
|Old/New Feature|old、new、valid、valid、valid|D/P位置未提供，后两语义槽仅常量|
|Full Role Feature|old、new、D、P、valid|完整Oracle特征|

Constant不会通过valid意外获得真值边界。其余三组共享同一有效域，缺失通道以
无信息常量占位，结构/可训练参数数量相同。所有方法都是显式研究上界；Constant
的oracle标签只用于来源验证，不进入特征值。不能把GT条件描述为可部署预测器。

## FLUX融合与兼容

只在显式特征上下文内，对Transformer输入的T=0噪声tokens加入编码残差；不新增
attention图像流、不改变source tokens/文本/position IDs，特征不通过Image VAE。
按实际四维(T,H,W,L) IDs gather，验证唯一完整网格、像素stride和范围，不能仅靠
token序号猜位置。固定版本BFL训练与Diffusers推理分别适配x/x_ids、hidden_states/
img_ids。四组均1024特征tokens、1024 source参考和1024 noise tokens，相同计算结构。

最后一层零初始化保证初始残差为0；禁用时不安装hook，保持原No-mask/Mask路径。
全零特征在训练后仍产生零残差；这不等于全P身份已被模型学会。身份行为需实测。
训练沿用固定ai-toolkit的LoRA、flowmatch/sigmoid目标；只通过已有extra-params hook
将编码器加入同一个optimizer，原loss不改。首步零输出层使上游特征层梯度可为0，
第二步后须出现有限非零梯度，否则不能宣布学习成功。

编码器单独FP32 safetensors与LoRA成对保存，绑定schema/配置/mode/base revision/
LoRA实际SHA。缺失、错配、非有限、格式或尺寸不符必须拒绝，不能静默回退。最小
试验只支持新目录独立训练及最终checkpoint重载，不承诺中断续训或多机DDP。

## 预注册预算与验收

新根outputs/role_feature_20261022。四组、训练seed211/223各20updates、batch4、
rank16、LoRA lr1e-4，编码器lr1e-3。每组从同一Base独立开始，特征网络/初始参数
一致，样本、token数、更新数与优化配方保持一致；旧RGB/灰度预算只作历史背景。
Dev固定12例、20steps/CFG4/ID seed42，逐seed、逐场景统计T target误差、D必要
变化误差、P source漂移、身份误差。重复物体和目标位置单列代理诊断与非盲案例，
不能拿像素指标当真人成功率；原真人包继续保留，真实评分仍pending。

先CPU契约/几何/位置/禁用与零残差/梯度/checkpoint测试；再单卡真实2-step/batch1
训练，重载前向/反向/有限非零梯度、角色置换、空间移动、零条件等价、重复PNG及
checkpoint重载测试。最后测batch4容量，记录实际时间/峰值，最多四卡并行。所有
GPU通过实时空闲检查与UUID租约。失败保留原traceback，用新目录重试，不放宽门禁。

正式比较还校验每个update的实际样本ID、noisy latent SHA与timestep轨迹，以及初始
特征权重SHA、可训练参数数、单source/特征token数；仅配方相同不足以证明公平。

工程门槛通过后才运行上述有界20-step。研究继续的必要条件：Full在两个seed的
非零Dev T/D均优于Constant和Binary，P不恶化，身份误差无实质恶化（预定容差1e-4），
并对Old/New展示额外收益；固定案例无明显重复/错误位置退化。四场景CI仅是固定
训练seed的场景不确定性，配对bootstrap2000/seed42，不作总体显著性或训练方差主张。
门槛失败为当前注入/预算No-Go；通过也只给provisional Go，真人未完成前不扩大训练。

最终执行Ruff/Pytest/全部CLI、真实GPU smoke、版本/文件SHA及提交资产审计；
记录已实现/已运行/负面/剩余风险。新实验必须有冻结代码、配置、数据与GPU provenance。

2026-10-22实现收尾时：73ba397冻结执行代码，CPU与CI验收通过；当日GPU因无空闲
UUID租约pending。此为历史资源状态，后续真实验收见[实际报告](role_feature_report.md)。

## 2026-10-23继续执行

用户确认GPU空闲后，实时查询8卡0MiB/0%、无计算进程。沿用冻结73ba397与原准备
目录，先执行尚未启动的2-step/batch1，再真实4B feature_smoke、2-step/batch4容量，
最后四方法×两seed×20updates及固定12 Dev。没有增加数据或训练预算。
新会话资源/验证/失败日志放outputs/role_feature_20261023；原准备目录中的空白任务
产物位置首次使用，已有资源失败日志、历史模型/数据/结果不覆盖。若发现兼容问题，
保留失败目录，在新目录重试，先写修复原因与回归要求，再修改并冻结新的执行提交。
真人缺失仍pending，Test只校验完整性；不开发Predictor、Qwen或新方法。

首轮GPU在extra-params初始化失败：固定上游TrainConfig没有seed属性（0个batch）。
兼容修复只将worker已经读取并用于set_seed的YAML seed显式传给本项目适配器，
不改上游配置类、loss或网络。回归必须使用没有seed属性的上游形状fixture，验证
211/223分别初始化、同seed一致、CPU RNG恢复及原失败清理；重试必须新目录。

修复提交b256692已冻结于role-feature-run-02，所有重试和正式实验在新study_01；
没有覆盖73ba397的原失败。231项CPU与43项CLI、真实4B探针、batch4及8组20-step
已实际通过。两seed各自四方法的20个实际update样本/噪声/timestep完全一致，
13,312 encoder/23,114,752总参数、1024特征及单source tokens符合预注册。
数据、预算、研究门槛未调整；质量判断独立于工程验收，真人评分不由程序填充。

真实96例推理完成后的首次CPU评估暴露身份字段对接错误：role_metrics没有
identity_mae，identity_target_mae表示source与target的距离，也不能当预测误差。
修复应在s=0子集读取已有target_mae（此时target=source），保持预定1e-4容差与
T/D/P门槛不变；测试必须接入真实region_metrics/aggregate输出，覆盖预测破坏身份
而source=target误差仍为0的情况。保留analysis_01失败，用新评估目录和源码快照
重算；不重训、不重新生成预测、不改原始指标或实验预算。
