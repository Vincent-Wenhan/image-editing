# Independent Role Feature Conditioning：机制诊断实际报告

2026-10-23 UTC。PR #5合并审计、Phase A CPU与真实GPU诊断已完成；最小改进及
公平对照尚未运行，不能宣称改进有效。旧受限No-Go
与全部历史资产保留，见role_feature_report.md。

PR #5正常merge为main 676242b，main CI39892792936通过。合并前本机232 tests/
192.17s、Ruff、43 CLI、23模型文件实际SHA/80.31s通过；96例指标/区间与实际
checkpoint复核支持原No-Go。5个全样本归约值仅约1e-17差，不影响门槛。
证据在原role-feature worktree的outputs/pr5_merge_audit_20261023。

已实现字段交换/消融/零填充移动/固定置乱，观测真实BF16融合、x_embedder后差，
按时间步/正负CFG分支缓存并固定状态回放，默认原路径/权重/损失不变。
相关14项/136.56s通过，包括小尺寸真实Diffusers Transformer（不是4B GPU）；
新增协议/资产漂移与数值回归日志另存probe_guard_tests.log。
新输出仅outputs/role_feature_diagnostics_20261023，新增训练0步；Test封存，
真人0评分/pending，不自动填写。

Phase A真实CPU测量已完成（encoder_analysis_01.json，9.20s，执行源码dcc9a1e）：
所有encoder都重构原seed初始化并核对原始SHA，参数确有更新，非“梯度没到”。
两个Full seed的类别/位置输入RMS比.06614/.06932，identity参考全局响应仅占原残差
RMS的.7042%/.5258%；原残差RMS为.00607766/.00594360。这是弱角色变化的线索，
还需校正编辑面积、数值分辨性及时间步，不是已确认的唯一原因。

首次正常GPU租约被8卡计算进程/约44GiB占用拒绝，未加载模型，不是训练/OOM失败。
保留study_01/smoke_211_execution及原trace；attempt重试输出隔离，成功任务复用。
后续attempt02正常租约成功：smoke127.30s，两seed完整诊断527.55/527.45s，
峰值allocated15.594/reserved16.299GiB，执行源码d35b4c1。6个原20-step PNG精确
复现、真实轨迹固定状态预测回放精确、零条件/禁用等价均通过。

CPU面积校正后identity差分在变化token内相对RMS为2.854%/2.211%；真实BF16
Old/New交换差分约91%–95%被舍入。FP32相加再cast改善极小，未发现CFG公共项
抵消的有力证据。具体最小修复预注册见role_feature_repair_design.md；尚未验证。

位置模板仅用已知source/target/background校准，低对比/参考定位失败或无匹配弃权；
重复模板候选不等于物体计数。D分阴影创建/移除和有向变化，P分边缘保持。
CPU增加变化token内响应与单位高斯BF16边界测量，明确不是实际GPU扩散输入。
相关数值/重试/语义/角色回归15 passed/13.79s，所有原始日志保留。
