# 角色条件研究验证计划（2026-10-22）

归档名称：**Causal Role Data and Oracle Feasibility Study**。以下保留原预定协议；
实际执行和当前编码 No-Go 见实际报告，不把后续失败分析混入本轮。

配置收尾补充：将scene sweep的物理采样范围独立保存为YAML，支持CLI选择；默认
范围必须复现本轮v1已经冻结的物理参数，不重新渲染、不改动现有数据或GPU任务。

本轮在 `feat/role-data-and-evaluation` 收尾研究工具并运行有界上界实验，不调整主要
目录，不训练 Role Predictor，不迁移 Qwen。历史 Stage 0/1/2 输出只读保留。
已阅读 Stage 1 补充报告、Stage 2 报告、proposal、架构与相关实现/测试。
以下是预先约定；完成状态和实际数字另写实际报告，不能把计划当结果。

## 真人评估

沿用匿名 task ID 与 source/instruction/target/prediction。每个导入文件只能包含一个
真实评审者，导入时记录其独立评分声明；至少三位不同代号、每人同一完整任务集。
程序不能证明不同代号对应不同真人，组织者须保证独立、未交流且未看私有方法映射。
原始 CSV 按原字节复制并记录 SHA256，不改写、不覆盖；修订另导入新目录，合并时
每个 reviewer 只能选一个版本，所有历史版本保留。旧单 CSV score 命令继续可用。

合并分别报告编辑成功度和无关内容保留（0/1/2/NA），不得用像素 MAE 代替前者。
比较在相同 reviewer × sample 上配对；先按 sample 聚合 reviewer，再按 scene/session
聚合其强度点，使每个独立场景等权。95% percentile bootstrap 同时对 reviewer 和
scene 有放回抽样，固定统计 seed；两个方法的配对关系保持。少于三位完整、独立
声明的 reviewer 时，整体状态 pending，推断区间为 null。NA 不当作零分，报告覆盖。
按预先标注的编辑任务分组；失败类别来自真实原始评分，多类别结论仍是描述性分析。
测试中的模拟评分明确标为 fixture，不写入任何实际人工结果。

现有180项全空人工表不填充。为三位评审者提供匿名、不同顺序的公开页面，方法映射
仍留 private；不通过工具联系他人。缺失真人结果必须 pending。

## Stage 1 公平性

复核 Constant / Static / Dilated 的实际 config、report、checkpoint、manifest、
控制索引、预测 SHA 与推理预算。相同双参考、同 VAE、BF16 cast 在归一化之前、
token 顺序、rank/alpha、seed、更新数和 batch。No-mask 原预处理与 training cast
保留为单独消融。不同参考数量的 FLOPs 不相等，不声称严格等算力。
为全部固定30例声明指令任务类别（人工策展的任务描述，不是成功/失败评分），分别
汇总 target MAE / mask 外漂移 proxy / 变化面积；人工任务与失败统计在无评分时 pending。

## 场景变化与独立分割

新增确定性程序化配置展开入口，控制 seed、split 组数、形状、宽度、颜色、物体初始
位置、点光位置/能量、棋盘背景参数与多个强度。展开后保存显式 YAML，无隐式随机
渲染默认值；不同 scene group 的所有强度仅在一个 split。支持 train/dev/test，旧
train/dev 配置仍有效。拒绝物理配置重复、跨 split session/source SHA 重叠和坏标签。

先在128分辨率验证三分割变化，再用相同冻结场景逐步验证512。维持旧阴影反事实、
逐轮廓 ignore、线性阈值、背景显露门禁；失败不放宽阈值、不覆盖输出，记录并修复。
这是简单 cube/cylinder、直接光硬阴影家族的 held-out 组合，不声称未见几何家族泛化。

## 四组联合条件

所有方法都为 `[source, condition_RGB]`，同 VAE、顺序、尺寸、训练预处理、LoRA
容量和原 ai-toolkit flowmatch/sigmoid objective，不新增 loss、网络或 mask 混合。

| 方法 | 第二参考 | 推理权限 |
|---|---|---|
| Static | source occupancy 的三通道二值图 | 常规 |
| Dilated | 同粗 mask、固定 r16 方形膨胀 | 常规 |
| Dynamic Binary | 当前强度的 valid ∩ (T∪D)，三通道二值 | 显式 oracle upper bound |
| Oracle Role | T红/D绿/P蓝单位向量，ignore黑 | 显式 oracle upper bound |

Dynamic Binary 也依赖真值，不是 predictor 或可部署方法。Oracle 与 Dynamic Binary
导出仅接受通过反事实重算的合成数据，索引保存角色、valid、source、evidence 摘要。
推理条件读取只打开 source/角色/valid/条件，不打开 target 或反事实 RGB。训练才加载
target。普通 infer/edit 拒绝 oracle；显式上界开关、用途和来源同时满足才能调用。
训练实际 batch 区分 RGB 二值 mask 与 T/D/P one-hot，检查有限值、几何和通道和。

## 预定预算与验收

新根 `outputs/role_validation_20261022/`，子目录全部禁止覆盖。工程先补单元/回归，
通过后冻结可执行 commit；逐 worker 保存源码摘要、配置、依赖、数据、seed 和租约。
先单卡 Oracle 2-step/batch1 与重载敏感性检查，测吞吐/显存，再测 batch4；足够余量
才并行，同一比较不自动扩大 batch。任何 OOM/失败保留并以新目录重试。

有界研究预算：12 train /4 dev /4 test scene groups，强度 [0,0.5,1]，512²；四方法
各两个训练 seed 42/43，60 updates、batch4、rank/alpha16、lr1e-4，均从同 Base
独立初始化，不续训 debug。按主机 RAM 容量限制并发最多四个 resident 模型。
dev 先完成固定全部12例、20 steps/CFG4/ID-stable seed42的重载评估。测试集仅在
配置冻结后运行一次同预算评估，不据测试结果调整方法。若资源不足，明确 pending。

自动指标分别报告有效 T/D 上的 target 误差与有效 P 上的 source 漂移、身份结果、
各强度和 scene 分组，以及原始 source 不编辑的 identity 参考；它们不是语义成功率。
配对差异按独立 scene 聚合，报告种子逐项和场景 bootstrap 区间。人工结论独立 pending。
正向初筛要求两训练 seed 的 Oracle 在非零强度 dev 上同时优于 Static/Dilated，且
相对 Dynamic Binary 有额外收益、P 漂移无恶化；否则不推进 Predictor。本轮数据/预算
很小，即使过门槛也仅建议继续扩大上界验证，不宣布研究成功。最终 Go/No-Go 随实际
结果给出，接口 smoke 仅证明条件消费。

验收：Ruff check/format、Pytest、全部 CLI/嵌套入口、安装/轻导入、真实 Blender
128→512、实际双参考训练/重载、条件置换敏感性、资产/第三方卫生检查；分阶段
Conventional Commits 并正常 push。实际时间、峰值显存、checkpoint、失败写入报告。
