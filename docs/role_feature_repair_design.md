# 冻结权重的角色差分干预：预注册

2026-10-23 UTC，Phase A真实运行完成后、Phase B实现与图像生成之前登记。
这是一个推理算子试验，不新增模型、不训练、不改loss或默认推理。沿用
role_feature_diagnostics_design.md的全部预算、Dev和Go/No-Go门槛。

## 选择依据与限制

真实4B smoke 127.30s，两个seed诊断各527.55/527.45s，峰值allocated15.594GiB、
reserved16.299GiB；执行源码d35b4c1。全部6个原始20-step PNG与旧实验精确一致，
固定状态原始预测回放与零/禁用等价检查通过。identity差分在变化token内仍只有
原残差RMS的2.85%/2.21%，不是仅由编辑面积小导致；Old/New交换差异在真实BF16
输入中约91%–95%被舍入为相同值。原残差/噪声约0.6%–1.0%。

仅FP32求和再cast的误差与原路径几乎相同；正/负CFG差分夹角没有支持“公共项被
CFG抵消”的证据。位移会改变valid边界，不能当作纯角色空间语义证据。非零输出
敏感性不等于学会角色语义；20updates、条件语义未充分学习也可能限制质量。

选择一次组合干预：去掉identity公共分量，并在原全局残差能量预算内放大角色差分。
不试多个增益、不搜索时间表。不能据此分离“去公共项”和“放大”的单独因果作用。

## 唯一候选算子

F含old/new/D/P/valid；identity(F)保留old，new=old、D=0、P=valid。
冻结同一encoder E，R=E(F)，R0=E(identity(F))，d=R-R0。

候选为 `clamp(d * RMS(R) / max(RMS(d), float32_eps), -c, c)`，
c沿用checkpoint的max_residual；RMS按每个样本全部token和通道计算。
RMS(d)≤float32_eps时严格返回零。不用target RGB、像素误差或质量指标选增益。
全P身份条件F==identity(F)时候选为零；这不保证生成器输出等于source。

四臂：original、zero、centered（上述候选）、centered_shuffled。
置乱只在完全相同的pooled-valid值内交换候选残差token，seed沿用20261023，
保留valid门控、残差值集合和能量，破坏空间对应；不沿用Phase A有valid混淆的置乱。

所有臂每个样本均预计算R/R0各一次，并用同一个融合hook缓存注入，核验original
缓存路径与旧PNG精确相同。每臂参数13312不变、1024特征token、单source参考，
相同20steps/CFG4/ID-stable seed42、相同Full checkpoint211/223、训练更新0。
不增加注意力流或参数；记录两次encoder调用及实际缓存/推理时间。

## 验证与输出

CPU先验证公式、零/身份、有限/上界、有效区置乱、参数/RNG、token/位置拒绝和
条件摘要篡改；GPU先4-step、身份与非零编辑各一例：原缓存/实时encoder等价、
身份候选/零条件等价、非零候选生效、有限值及重载重复。通过后两seed×12Dev×
四臂共96张，沿用原训练权重，训练预算均0。实际GPU启动前查进程/空闲并正常租约。

保存实际FP32原始/identity/候选残差、字段与置乱索引的文件SHA及参数/数据SHA，
不上传Git；CPU复核公式与文件，跨CPU/GPU重算只使用明确浮点容差而不伪称逐位一致。
输出使用新目录，保留所有失败；Test只做完整性校验，不推理、不选型。

沿用预定两seed各T/D改善≥1%、P/身份增量≤1e-4及zero/置乱对照门槛，场景等权、
2000次bootstrap/seed42。分列D创建/移除、有向变化、P边缘与位置代理的弃权覆盖。
非盲画廊单列，不作为真人评分。人工包保持空白，真实评分pending。
未通过则当前冻结权重干预No-Go，不增加数据/训练步数；T/D/P本身的价值另行限定讨论。
