# 评估完整性修复与角色条件下一轮方向设计

2026-11-05 Asia/Shanghai。本文先于本轮代码修改登记范围、验收方式和研究边界。

## 本轮范围

第一部分修复评估产物完整性：画廊必须复用推理 provenance 校验，拒绝失败或不完整
run，拒绝已有输出目录，所有 manifest 路径必须经过数据根目录防逃逸检查，并支持
普通 mask、`oracle_role` 和 `oracle_geometry` 三类控制。预测图必须保持 RGB 和记录尺寸。

第二部分补强实验 provenance：执行记录绑定项目依赖锁、环境文件、脚本和外部仓库的
实际状态，包版本记录覆盖训练/推理/评估使用的主要运行时依赖。封存 cohort 消费端
校验完整的 Dev/Test 声明，避免只依赖部分哈希字段。

第三部分记录下一轮研究方向，但不在本轮直接扩大同一残差注入训练。已有原始注入与
`channel_rms` 均未通过角色方向门槛；下一轮候选改为显式几何方向条件：在 Oracle
诊断中增加由 old/new 占据推导的有向位移场和边界距离场，作为独立条件输入，与现有
old/new/D/P/valid 做同预算、同噪声、同 checkpoint 的对照。该条件只作为合成数据的
机制上界，不进入普通推理，也不替代未来的可部署预测器。

候选方向的 Go 条件是：CPU 契约、输入来源、空间对齐、identity/zero、梯度和重载
门禁全部通过；之后在新 Dev 上预注册纯噪声与中低时间的 T/D 正确方向、P/identity
保持、zero/shuffle 对照。未通过时停止该方向，不继续调增益、时间表或训练步数。

## 不在本轮做的事

- 不修改外部 ai-toolkit、Diffusers 或原 flow-matching loss。
- 不读取 target RGB、监督 target 或 Test 作为推理输入和条件选择依据。
- 不启动新的 GPU 训练；GPU 任务必须等待 `nvidia-smi` 显示符合门禁的空闲卡。
- 不把像素代理、非盲观察或 loss 下降写成人工成功率或泛化结论。

## 验收方式

- `pytest -q` 在完整项目环境通过，新增回归覆盖失败/不完整 gallery、已有输出拒绝、
  Oracle 控制渲染、路径防逃逸、预测模式检查和 provenance 变更拒绝。
- Ruff check/format、`compileall`、CLI help、仓库卫生检查通过。
- provenance 的旧字段仍可读取；新增字段缺失时只保留明确的兼容降级，不静默放宽
  当前新实验的完整校验。
- 研究方向只完成契约、编码、CPU 测试和文档登记；质量结论留到单独预注册 GPU 实验。

## 本轮实际验收

284 项本机测试通过，新增方向和画廊回归包含在内；Ruff check/format、compileall、CLI
help、pip check 和仓库卫生检查通过。由于当前所有 GPU 均不满足空闲门禁，本轮没有启动
新的训练或推理，也没有产生新的模型质量结论。
