# T/D/P 可学习性：实际状态

2026-10-23，Asia/Shanghai。PR #6已正常合并main 8b261bb；新分支feat/role-learnability。
合并复核252 tests/218.67s、55 CLI、96图实际指标/配对区间及空白盲评包通过，
证据保存在role-feature-diagnostics worktree的outputs/pr6_merge_audit_20261023。
历史No-Go和全部数据、checkpoint、失败/真人包保留。

本阶段已预注册并实现两Train场景6例、四方法双seed各120updates、共享中性指令
与同组初始噪声、梯度/层观测。17项相关CPU测试102.83s通过。
首次真实GPU smoke冻结d6026cd，在AcceleratedOptimizer注册step hook时报
AttributeError，0个batch/0训练更新；初始快照、traceback和显存报告
保留study_01，不是模型质量结果。自有代码将hook注册到包装器内部optimizer，
补真实继承布局回归后用study_02重试。第二次完成2个batch/更新（211.53s，
peak allocated15.022/reserved15.037GiB），但上游post_save_hook传入optimizer.pt，
首次encoder错误绑定optimizer，最终重复路径被拒。两次失败快照的该绑定均无效，
不得作为有效模型对；LoRA/无效encoder/trace仍保留。改为观测network.save_weights
的实际LoRA路径并补回归，使用study_03重试，不修改第三方或覆盖旧目录。
当前可学习性/注入改进均未验证；真人0条/pending，Test继续封存。
