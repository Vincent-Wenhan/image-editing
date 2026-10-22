# 角色研究验证实际报告

正式研究名称：**Causal Role Data and Oracle Feasibility Study**。
合并前独立复核结果见 [审计记录](causal_role_feasibility_audit.md)；本报告原始结果不改写。

更新时间：2026-10-22（Asia/Shanghai）。结论：**当前彩色 Oracle Role 路线 No-Go，暂停扩大训练及 Role Predictor。** 两个训练 seed 的 dev 门槛均未通过，独立 test 也没有支持收益。Dynamic Binary 的分区像素误差较低，但有重复物体，真人质量仍 pending。

本轮根目录 `/home/jiangwenhan/project/image-editing-role-data/outputs/role_validation_20261022/`。旧 Stage 0/1/2 的图像、评分、权重、失败目录全部保留；本轮使用新目录，资产不入 Git。预定协议见 [设计](role_validation_design.md)。

## 场景、标签与分割

固定采样 seed=20261022，12 train /4 dev /4 test 独立物理 scene groups，每组强度 0/0.5/1，共36/12/12对。程序化变化宽度、颜色、初始位置、点光位置/能量和背景棋盘；同组所有强度只在一个 split，各 split 两两 session/source SHA 无重叠。仍是 cube/cylinder、固定相机、直接光硬阴影家族的组合，不能外推到未见几何家族或复杂物理。

| 分辨率 | 渲染 worker (s) | 完整生成/校验 (s) | 背景显露最大线性误差 |
|---|---:|---:|---:|
|128²|41.7191|48.4411|0.00630373|
|512²|396.6238|473.4204|0.00743136|

两批物理配置相同，均通过配置/EXR/NPZ/五类PNG实际SHA、反事实角色重算、独立旧/新轮廓ignore、valid、位移与背景补全检查；保持原0.01门禁及角色阈值。每批20个s=0身份对精确source=target、全P，40个非零对均有有效T/D/P。`scene_gallery/` 展示真实标签。

实际数据由 physical_sweep_v1 生成，源码摘要留在 generation.json。随后将相同范围抽到 `configs/datasets/blender_variations.yaml`，新v2保存完整范围快照、固定split顺序；默认展开的20个物理场景与v1逐项完全一致。没有重新渲染本轮或更换训练数据。

## 模型、条件与预算

固定未蒸馏 FLUX.2 Klein Base 4B revision `a3b4f4849157f664bdbc776fd7453c2783562f4d`，ai-toolkit `a055947d56a156411abb4ef0301b35d0fe5f51a5`；本地VAE/TE和模型权重复用，没有重新下载或修改第三方源码。

四组均为 `[source, condition_RGB]`，512²、同VAE/先BF16 cast再归一化、同参考顺序和token数量、rank/alpha16、lr1e-4、batch4、60updates，训练seed42/43，均从Base独立开始。流程仍为原flowmatch/sigmoid objective，没有新增loss或网络。完整配方与数据SHA在 `matched_budget_42.json`、`matched_budget_43.json` 均验证通过；每任务独立数据副本隔离缓存。

|方法|条件与权限|
|---|---|
|Static|source可见占据三通道二值图，常规推理|
|Dilated|同mask、预定r16方形膨胀，常规推理|
|Dynamic Binary|valid内T∪D，依赖真值，仅训练或显式上界|
|Oracle Role|T红/D绿/P蓝one-hot、ignore黑，仅训练或显式上界|

条件消费只读source/labels/valid/控制图，不打开target或反事实RGB；普通edit/infer拒绝oracle-derived条件。真实训练batch检查one-hot/二值、几何、范围和双图；每个正式checkpoint核验160个有限LoRA张量、实际SHA、至少60次联合条件检查。

GPU执行冻结在 `/home/jiangwenhan/project/image-editing-role-validation-run` commit `6879c87`。8个训练worker和16个推理subprocess均记录相同冻结源码摘要、argv、依赖和GPU UUID；每个execution完成时再次检查代码指纹。CPU最终统计在 `878c911` 执行（`analysis_execution_01`），后续门禁/统计和采样器配置不改变冻结的模型计算。

## 最低GPU验收与真实耗时

均先实时检查空闲卡并通过UUID租约启动，先单卡测量再四卡并行，保持固定研究batch。

|验收|worker (s)|含启动完整wall (s)|peak allocated / reserved (GiB)|
|---|---:|---:|---:|
|Oracle 2-step /batch1|143.7247|151.9945|15.0219 /15.0371|
|重载、T/D置换、重复采样|71.0070|103.9591|15.6121 /16.3945|
|Oracle 2-step /batch4|158.4285|166.7847|15.0219 /17.0859|

重载真实VAE编码恰好两图，角色输入仅-1/+1且最多一个正通道，attention T=0/10/20各1024tokens；T/D置换改变控制token和输出，source/noise保持，重复PNG逐字节一致。这个检查证明条件消费和可重复性，质量由后续对照判断。

最小checkpoint SHA：`7fd99d95c42ffc6a069cb9e98a9bab4c02c59a66f80f9b36afbe4211df947ac6`；容量checkpoint SHA：`24a92347ed1efc3b18b8c1c2fd294d0c861aee4429a69bebe6fc2fc8b004a390`。目录分别为 `study_01/smoke_oracle_role_42/`、`capacity_oracle_role_42/`。

正式训练peak allocated/reserved均15.0219/17.0859GiB。每组推理dev/test各12例，20steps/CFG4、逐ID稳定seed42，全部192张完成；推理peak reserved均16.3945GiB。下表推理为每12图的完整进程wall，包含加载与租约开销；逐图生成时间另在run.json，汇总于execution_summary.json。

|seed|方法|训练worker (s)|dev推理wall (s)|test推理wall (s)|
|---:|---|---:|---:|---:|
|42|static_mask|592.85|295.76|283.71|
|42|dilated_mask|593.66|293.81|277.16|
|42|dynamic_binary|598.97|295.63|286.29|
|42|oracle_role|607.25|296.31|282.37|
|43|static_mask|579.95|276.21|274.91|
|43|dilated_mask|589.00|279.15|274.93|
|43|dynamic_binary|579.70|281.63|284.12|
|43|oracle_role|570.32|279.35|281.95|

所有正式checkpoint位于 `study_01/train_<method>_<seed>/lora/train_<method>_<seed>/train_<method>_<seed>.safetensors`。完整SHA如下（可与checkpoints_verified.json核对）：

```text
42 static_mask: 47867da63f85a07eb85dff81d4b4541d974b2ff8c90ef6e311d99ae672832d82
42 dilated_mask: 8f7b869c8b922d1a4b6b01a518ea0b94d8c9d6ed2628006282dbb46bfba8a500
42 dynamic_binary: 1a0d903ad9bc34c3e9bf807efc8b057852582f131d978611bc6ea1c957413e0e
42 oracle_role: 8ae3c53c374afa118cf7bdd0d5f6f71acb85ebc0360d3b94b12e2a2825153a24
43 static_mask: 0b27542b56f545cf5562f4e7c283c9cc0d2c0a93f986f4f93560c6f991f4f34b
43 dilated_mask: 2ff1ed32cbf740e825a76d44b027b3169c629ad178172fb34ee1e467fac6825c
43 dynamic_binary: 7545a4d227d673d9eb4b27b07c56c2f8f42db242983a4c49d2ec78ed417a6b06
43 oracle_role: 237741c501e90020537bca02fba9279faeb113ff7d9a5d400dbae98269e95f75
```

seed42对Static逐张量比较：Dilated/Dynamic Binary/Oracle均160/160张量有差异，最大绝对差分别0.002476/0.002701/0.003277。记录在adapter_tensor_comparison_seed42.json，只是参数敏感性诊断；条件消费另有实际VAE/attention干预证据。

## 分区结果与配对统计

以下为非零强度8例/4独立scene的等scene权重均值：T/D使用target MAE，P使用source漂移，均越低越好。**这些不是语义编辑成功率或人工保留评分。** s=0独立报告，不事后筛选成功样本。test在方法/预算冻结后只运行一次，不据结果调参。

|split|seed|方法|T target MAE|D target MAE|P source drift|
|---|---:|---|---:|---:|---:|
|dev|42|static_mask|0.116827|0.281596|0.010165|
|dev|42|dilated_mask|0.107932|0.272976|0.012196|
|dev|42|dynamic_binary|0.090875|0.121226|0.008812|
|dev|42|oracle_role|0.361112|0.195494|0.010365|
|dev|43|static_mask|0.132010|0.315238|0.011436|
|dev|43|dilated_mask|0.109085|0.289386|0.012675|
|dev|43|dynamic_binary|0.094458|0.144549|0.009046|
|dev|43|oracle_role|0.301835|0.219948|0.010458|
|test|42|static_mask|0.136847|0.302669|0.010401|
|test|42|dilated_mask|0.101959|0.226827|0.010609|
|test|42|dynamic_binary|0.088098|0.081927|0.008430|
|test|42|oracle_role|0.304732|0.235611|0.016039|
|test|43|static_mask|0.131204|0.301333|0.015044|
|test|43|dilated_mask|0.128246|0.238644|0.014116|
|test|43|dynamic_binary|0.081103|0.082063|0.008936|
|test|43|oracle_role|0.274173|0.166799|0.016942|

不编辑source的T/D误差参考：dev=0.149224/0.394674，test=0.129277/0.350236；不是训练过的No-mask模型。Oracle的T误差在两seed、两个split均高于identity参考。

Oracle−Dynamic Binary的配对95% percentile区间，先聚合同scene强度，再以scene为单位bootstrap2000次、seed42。正值表示Oracle误差较大。每split只有4组，区间是固定训练seed下的场景不确定性，不代表训练seed总体方差，也未作多重比较校正。

|split|seed|T差值 [95%CI]|D差值 [95%CI]|P差值 [95%CI]|
|---|---:|---|---|---|
|dev|42|0.270237 [0.206441, 0.334034]|0.074268 [0.003044, 0.154553]|0.001553 [0.000188, 0.003478]|
|dev|43|0.207377 [0.160259, 0.254496]|0.075399 [0.049783, 0.101016]|0.001412 [-0.000713, 0.004322]|
|test|42|0.216634 [0.167877, 0.264211]|0.153683 [0.106781, 0.192588]|0.007609 [0.005538, 0.009542]|
|test|43|0.193070 [0.170307, 0.222342]|0.084736 [0.038963, 0.125417]|0.008006 [0.004868, 0.011267]|

完整逐样本、各强度、全样本/非零均值和所有方法配对在 `analysis_01/*.csv`、`summary.json`；决策门槛在go_no_go.json。s=0 dev的Oracle P漂移为seed42 0.039526 /seed43 0.098407，Static为0.010611/0.012153；身份行为也需要修复。

## 编辑效果与失败案例

以下为代理对**预定前两个dev组、全部强度**的非盲目视观察，不是三人评分，不计算人工失败率。联系图为 `dev42_diagnostics/fixed_cases.jpg` 和 `dev43_diagnostics/fixed_cases.jpg`，选择与原图SHA留存。

|样本|观察|
|---|---|
|dev_20261022_000_s001 /s002|Oracle复制红/绿控制图外观，部分生成两个红块；Dynamic Binary在大位移时重复青色物体；Static/Dilated的位置或形状仍偏离。|
|dev_20261022_001_s001 /s002|Oracle生成红色圆柱或保留旧绿圆柱再添加红柱；Dynamic Binary有重复圆柱；Static/Dilated的位移/大小未匹配目标。|
|seed43 /dev_20261022_001_s000|正确的全P身份条件仍出现明显蓝色背景块，破坏原本无需改变的区域。|

观察与把角色RGB当普通参考外观复制的解释相容，机制尚未单独验证。更低的Dynamic Binary像素误差也不能消除重复物体这一语义问题。

## 真人评审、检查与决策

Stage 1三组实际150-step/30例公平性与分组复核见 [补充报告](stage1_review_report.md)；最终证据为stage1/audit_02.json。Stage 1每人180任务，新增角色dev包每人96任务，均已准备三份独立顺序页面。真实评审者0位、原始评分0条，全部pending，所有人工均值/区间为null。没有用代理观察或fixture填评分。

导入/合并保留原字节和SHA，每个reviewer只选择一个历史版本，至少3位完整且独立声明的真实评审后才产生scene/reviewer交叉bootstrap区间；分别统计编辑成功度和无关内容保留，NA明确缺失，失败分类为结果筛选后的描述。组织者仍需保证实际真人独立性。入口在stage1/human_panels及role_human_review/panels，汇总pending路径分别为stage1/human_pending_02和role_human_review/pending。

本机202 passed/86.83s，Ruff check/format、全部CLI、依赖检查和仓库外wheel安装/配置展开通过（100个wheel条目，含范围YAML）。曾有一次CPU CI因模拟backend遗漏包版本stub失败（190 passed/7 skipped/2 failed），已修复并通过后续CI。训练、重载、容量、16个正式推理均无执行失败/OOM/cleanup错误；历史失败文件照常保留。最终GPU实时检查均0MiB/0%且无计算进程。

**No-Go：暂不继续扩大当前 `[source, one-hot role_RGB]` Oracle训练，不进入Role Predictor或Qwen迁移。** 两训练seed都未满足预定dev门槛，test同样显示Oracle在T/D上弱于Dynamic Binary，P没有稳定额外收益，还出现控制图颜色复制、重复物体和身份破坏。结论限定于当前编码、12训练场景组和60updates的小规模预算。

下一步保留现有全部结果，先完成真实三人盲评；用新的预注册预算研究编辑后端对控制图外观复制与旧/新占据歧义的处理，比较更简单的Dynamic Binary上界。任何新选择先在dev验证，并另保留新的独立test场景；待Oracle本身提供语义收益后再考虑预测器。当前不实施这些新方法。
