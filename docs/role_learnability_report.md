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
study_03冻结2f439d6：两步batch1训练成功198.78s，peak allocated15.022 /
reserved15.037GiB；GPU前反向/zero/空间对齐/重载成功105.33s，15.866/16.268GiB。
两步batch6容量检查成功266.36s，15.066/18.047GiB，低于总显存80%门槛。
实际checkpoint与encoder摘要、各通道/层梯度见study_03原始JSON。
正式四方法双seed各120updates已启动，启动时正常预检跳过占用的GPU0，先使用
空闲GPU1–4；本报告不将尚未完成的训练或诊断列为研究验证。
新增356图固定矩阵与监督/分层/梯度工具已实现，新增诊断GPU smoke和完整评估待运行。
新增诊断及语义代理相关20 tests /22.08s通过，Ruff通过；完整套件待最终执行。
当前可学习性/注入改进均未验证；真人0条/pending，Test继续封存。
