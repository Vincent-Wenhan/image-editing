# Independent Role Feature Conditioning：机制诊断实际报告

2026-10-23 UTC。当前仅完成PR #5合并审计、预注册和Phase A工具/CPU检查；本阶段
GPU诊断、机制结论、最小改进及公平对照未运行，不能宣称改进有效。旧受限No-Go
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
