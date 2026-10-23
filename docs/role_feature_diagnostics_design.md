# Independent Role Feature Conditioning：机制诊断预注册

2026-10-23 UTC。PR #5正常merge为main
`676242b17a5d3e1173b08e88e0e3606af80dbaf6`，新分支feat/role-feature-diagnostics。
合并前232 tests/192.17s、Ruff、43 CLI、23个模型文件SHA/80.31s及资产审计通过。
96例指标/区间与原No-Go重核；仅5个全样本归约值有约1e-17浮点差，门槛、数据和
checkpoint相同。原证据在role-feature worktree的pr5_merge_audit_20261023。
保留原报告、失败、权重、数据与真人包；不修改external。

## 范围

区分弱角色信号、BF16量化、时间步/CFG响应与角色/位置混淆。先测量，再选择一个
有依据的最小改进。新增训练更新固定为0，不开发新模型/编码器，不扩量，不做
Predictor/Qwen。改进仅改变显式诊断采样，不影响默认No-mask/Mask/特征推理。
这是已有权重的推理干预，不冒称重新训练的公平方法比较或否定全部T/D/P研究。

复用512²场景seed2026102201、原cohort seal，12/4/4 train/Dev/封存Test场景组。
只对Dev推理；Test仅完整性校验。Dev已反复使用，不称为未见泛化验证。
四方法×seed211/223原20-update checkpoint保持字节不变，不恢复optimizer。
新资产仅outputs/role_feature_diagnostics_20261023，失败保留，新目录重试。

## Phase A：机制测量

CPU核对实际encoder/LoRA/manifest/evidence SHA，分析四方法×两seed×全部12例：
残差RMS/Linf、token变化、类别/位置项大小、identity参考及逐通道删除的输出差、
D/old/new/P token覆盖，区分面积与单位作用。identity参考定义old不变、new=old、
D=0、P=valid，完全由条件构建；它不保证旧checkpoint已经学会身份保持。

单卡GPU smoke后，两个Full checkpoint各诊断三例：000_s000身份、000_s001
中强度cube、001_s002满强度cylinder。采样沿用20steps/CFG4/ID seed42。
original重采样核对原PNG，固定source/指令/初始噪声/预算。
干预：original、zero、identity、old/new交换、D/P交换、分别删old/new/D/P、
右移4个token（64px，零填充，valid同移）、固定seed空间置乱。它们是OOD敏感性
实验，不是训练过的独立方法。生成图单独作T/D/P、身份、位置、阴影和背景分析。

缓存原轨迹第0/4/9/14/19步注入前输入，含CFG正/负分支。每次干预使用**相同
状态**再前向，记录实际timestep，不拿不同生成轨迹差当局部敏感性。测量：

- 噪声token/残差RMS及比例、BF16加法保留率/量化误差、通道交换被舍入为相同
  输入的比例；FP32相加再cast仅作数值反事实。
- x_embedder后的原特征/实际条件差、Transformer速度预测差/相对RMS；CFG正/负
  与组合响应分列，不能把公共项描述为完全消失。
- old/new/D/P作用、identity/common空间分量、空间扰动；source/reference、文本与
  同状态输入SHA不变。全零应与禁用hook严格等价。

smoke先4-step单例，验证重复、零条件、有限值、位置与hook清理；正常UUID租约
最多两卡并行、模型驻留，启动前实时查空闲GPU，不中断他人任务。

## Phase B：一个最小改进

Phase A后先记录选择依据/替代解释、唯一改动和配置，再实现并做CPU/GPU smoke。
不预设BF16或位置项是主因，不同时试多个增益/时间表，不做Dev网格搜索。
只允许冻结权重的单一数值或残差变换，不增加参数/tokens；证据不足时如实限制因果结论。

两seed、全部12例Dev、相同source/prompt/20steps/CFG4/seed42，对照original、zero、
唯一改进和改进的固定空间置乱。训练0步、参数/tokens相同；记录checkpoint/条件/
输入SHA、时间、显存和PNG。置乱是结构对照，不是部署方法。不能仅凭去条件后
漂移减小就宣称角色语义收益。

**继续门槛**：两seed非零T和D相对original各改善至少1%，P及身份MAE增量均≤1e-4；
对zero及置乱，T/D均不更差且至少一项改善1%。固定案例无明显新增重复/位置/主体
破坏。scene等权、bootstrap2000/seed42/95%；四场景区间不是训练seed总体方差。
通过也仅provisional，真人未完成不扩量。失败为当前改进No-Go，不事后改指标/容差。

## 语义评估与验收

完善位置/重复候选、阴影变化、背景保留与叠图，使用已知source/target作校准和
弃权检查。模板/颜色/像素派生量明确为合成场景代理，不称真人成功率、可靠通用
物体计数或物理理解。T编辑、D必要变化与P保留分列；真人接口保留，程序不填分。

测试覆盖BF16测量、真实位置ID/参考不变、干预无target读取、固定状态/CFG配对、
seed/config/SHA、语义校准/弃权、checkpoint与失败清理。代码≤500行/文件，
src.xxx绝对导入，Torch懒加载，无sys.path补丁；Ruff/Pytest/CLI、真实GPU smoke、
冻结源码/版本与资产审计后按Conventional Commits正常提交推送。
本文是计划，实际主因证据、改进是否有效与研究建议另写实际报告。
