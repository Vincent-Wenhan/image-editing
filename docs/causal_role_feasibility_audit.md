# Causal Role Data and Oracle Feasibility Study：合并审计

日期：2026-10-22 UTC。审计对象 `feat/role-data-and-evaluation` / PR #3，代码
`6fe76ec`；只补归档名称和审计文字，不改模型、数据、统计协议或旧结果。

## 证据复核

本地新记录：`outputs/merge_audit_20261022/`（位于 role-data worktree）。

|声明|独立复核|证据|
|---|---|---|
|128/512 各20组、60对反事实数据|重新执行完整 SHA/角色重算/有效域/背景补全/split 校验，通过|`evidence.json`；原 `scenes128_01/`、`scenes512_01/`|
|最小、容量及8组正式训练|10个 checkpoint 实际 SHA 与完成报告一致；冻结版本一致|原 `study_01/*/train_report.json`，`evidence.json`|
|192张固定预测及分区指标|16个运行逐图 SHA、RGB/尺寸、数据与 adapter 绑定；从图像重新评分|`evidence.json`；原 `analysis_01/summary.json`|
|配对置信区间|从原逐样本记录重算所有方法/seed/split 的场景 bootstrap|最大数值差 2.78e-17，容差1e-12；不是新增实验|
|完整来源|32份训练/执行记录的逐文件摘要与其声明的 Git 对象逐一相符|`provenance.json`；GPU执行冻结 `6879c87`|
|真人评分 pending|两套实际汇总均无 reviewer，无真实评分|Stage 1及角色评审 pending/summary.json；未创建评分|
|无误提交资产|25个 Conventional Commits、202个历史blob，最大34,857 bytes；无禁止资产|`provenance.json`；第三方三个锁定仓库干净|

本轮复核 Ruff check/format、**202 passed /99.26s**、22个 CLI 帮助入口（含根入口）
及仓库卫生通过。复核前 PR 的 push/PR CI 均成功：
[push](https://github.com/Vincent-Wenhan/image-editing/actions/runs/39662320195)、
[PR](https://github.com/Vincent-Wenhan/image-editing/actions/runs/39662330575)。
归档提交之后仍须检查最新 head 的 CI 再正常 merge，不能用旧 head 的绿灯代替。

审计临时执行曾因误写一个未使用的检查函数名称退出、又因直接比较浮点聚合字典
的末位差异退出；纠正审计调用后完成上述复核。两次均未启动 GPU、未更改旧资产或
训练/推理结果。原始审计日志保留，没有将退出伪装成模型失败。

## 结论边界

原设计要求两个训练 seed 的 Oracle 在非零 Dev 上优于 Static/Dilated，并相对
Dynamic Binary 提供额外收益且 P 无恶化。两个 seed 都未满足，Test 同样不支持；
因此当前彩色 one-hot RGB Oracle **No-Go** 有记录支持。颜色复制、重复物体和
全 P 身份破坏属于原定样本的非盲代理观察，不是人工发生率，也未单独证明机制。

Dynamic Binary 是使用真值的**区域上界对照**，其较低像素误差不能证明位置正确或
没有重复物体。只否定当前编码/数据/预算，不否定所有角色条件方法。原 Test 已用于
评估和查看，不能再次声称未见；下一研究须使用新的 Dev，另封存新的独立 Test。
真人独立性和语义质量仍待实际评审。保持所有历史资产，只做正常 PR 合并，不 force push。
