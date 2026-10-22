# Role Conditioning Failure Analysis：实际报告

2026-10-22 UTC。PR #3 经最新CI及资产/统计复核，正常merge为main `a647522`；
前轮归档为 **Causal Role Data and Oracle Feasibility Study**，其彩色Oracle No-Go
及全部结果保留。本轮分支 `feat/role-conditioning-failure-analysis`，PR #4。
预注册见 [设计](role_failure_analysis_design.md)，旧研究审计见
[归档审计](causal_role_feasibility_audit.md)。

本地根 `/home/jiangwenhan/project/image-editing-role-failure/outputs/role_failure_20261022/`。
模型/第三方版本沿用前轮固定值，无重新下载模型、修改第三方或覆盖旧资产。
GPU/CPU诊断冻结在独立worktree `image-editing-role-failure-run` 的 `a07ba92`；
Blender生成在未再修改的 `d7b6414`，保存完整执行及生成器摘要。

## 数据与封存

新场景seed2026102201，512²，12 train/4 Dev/4 Test groups，每组强度0/0.5/1，
共36/12/12对。全量SHA、EXR/NPZ/PNG、反事实角色重算、valid、物理位移和背景
补全通过：完整生成452.329s，含启动455.826s；背景显露最大线性误差0.00600635，
原0.01门禁不变。与前轮20组和初始Stage2的全部split无group/source SHA重叠。

`cohort.json`固定dataset/各manifest的实际SHA和全部Dev IDs。新Test仅生成和完整性
校验；无模型推理、画廊、评分或模型选择。最小协议拒绝Test split，测试覆盖该门禁。
仍是固定相机、cube/cylinder、直接点光硬阴影家族，不代表未见几何或真实场景泛化。

```text
dataset: 1c1b2ddc7f5d1df5d9cd0bf11eee2c405d49a128f8745682737623ddf9631c21
train manifest: 690d3c864409c5d563c69652ef3e21946e119fec60d467ddc6d1c20862e2f8f9
dev manifest: b6778f19688023a6720041fe9f1a8b69fcd69e4b2d505b32784ee3a020efb29f
sealed test: 3a6f26bc9bb4896066046a5597d5fa7bea59c94ecfdbadc3f5decda1c6c8d68b
```

## 旧checkpoint诊断

复用原60-step、seed42/43的Oracle/Dynamic Binary四个checkpoint；先核对原训练报告
及实际权重SHA。每个模型新Dev完整12例baseline，再对预定前两个groups的全部6例
运行original与两种干预，共每worker30次生成，四worker合计120次。original重采样
与baseline逐字节一致；每次两个真实VAE参考、T=0/10/20，干预source/noise token
摘要保持。全部运行完成，无OOM/执行失败；详见diagnostic_phase.json及各执行日志。

所有采样20steps、CFG4、ID seed42、512²，同source/prompt/adapter/预算。配色或
删区域属于旧checkpoint的分布外干预，不能当作重新训练的公平方法对照。

|训练seed|条件|新Dev非零T target MAE|D target MAE|P source drift|s=0 P drift|
|---:|---|---:|---:|---:|---:|
|42|RGB Oracle|0.374745|0.261421|0.016111|0.014898|
|42|Dynamic Binary|0.082423|0.098535|0.009014|0.014412|
|43|RGB Oracle|0.317173|0.231838|0.015331|0.016653|
|43|Dynamic Binary|0.084187|0.090929|0.009682|0.015588|

每个非零均值8例/4独立scene、scene等权；不是编辑成功率。新增scene bootstrap
在new_dev_checkpoint_comparison.json，固定训练seed/4场景，不能估计训练seed总体
方差。逐样本、干预与实际PNG摘要见failure_analysis_01；CPU完整执行31.16s。

### 颜色与身份

循环RGB保持空间分区，模型生成色随之变：固定满强度cube的红色表面变绿、绿色侧块
变蓝；cylinder也生成绿色新物体。预定前两组的4个非零样本，T区输出色变化对控制
色变化的有向归一化响应，seed42=0.694801、seed43=0.537963（1为完全跟随，0无
同向变化）。这是像素敏感性，配色同时改变已训练的通道语义，不能排除标签含义
混淆，也不能单独证明全部机制。固定联系图与SHA在fixed_intervention_contacts_01。

P改黑不能修复该固定满强度样本的红绿复制。**新Dev四个身份场景、两个Oracle seed
均未目视复现历史的大蓝块**，P漂移如上；不能宣称严重身份破坏在本轮普遍重现。
历史seed43/dev_20261022_001_s000的蓝块图与记录原样保留，不将其混作新Dev实验。
identity_dev_preview_01覆盖全部四个身份场景。所有目视描述都是代理非盲观察，
没有写入任何真人评分。

但预定的身份换色干预检出另一种失败：seed43的dev_2026102201_001_s000真值全P，
将蓝色P参考循环为红色后，输出左侧出现大红边栏，P漂移从原0.010924升至0.077728；
P改黑为0.011731。source/noise/prompt保持，输出/控制实际SHA和全P检查另存
identity_palette_intervention_01.json。该红色参考是**分布外配色**，在训练编码中
对应另一通道含义，不能冒称合法蓝色P输入复现了旧大蓝块；它支持配色相关的身份
风险，同时保留外观复制/通道语义混淆两种解释。这不是新增样本选择或新增GPU运行。

### 区域上界、重复与位置

Dynamic Binary仍是依赖真值的**区域上界对照**，普通推理拒绝它。固定cube/cylinder
满位移样本均见旧物体残留并新增物体。仅旧占据会停留旧位置或扭曲形状，仅新占据
更接近目标但可产生背景块；两者都删去了D/另一占据，不能据此宣称新方法更好。

预定两组4个非零样本，原Binary的旧独占区source距离仅0.014853/0.012595，而
其target误差0.080868/0.081690；新区target误差0.090304/0.087899。两类区域单独
报告支持“旧区仍像source”的诊断，不能代替物体计数和真人目标位置判断。
T或T∪D不区分旧新，对调不变；完整输入仍含source与指令，这不证明任务不可识别。

## Geometry最小试验

已实现灰度旧/新独占64/192、重叠128、D255、P/ignore0；同双参考/VAE/原loss和
网络，普通推理拒绝该真值条件。推理只从实际SHA验证过的NPZ打开两个占据数组，
不读取target或反事实RGB；单元测试对此设访问守卫。真正独立特征encoder仅设计，
未实现、未训练。

|工程验收|worker(s)|完整wall(s)|peak allocated/reserved GiB|
|---|---:|---:|---:|
|Geometry2-step batch1|166.190|176.341|15.0219/15.0371|
|重载/旧新互换/重复|89.664|118.756|15.6121/16.3945|
|Geometry2-step batch4|180.640|191.934|15.0219/17.0859|

最小checkpoint SHA `40141ea9041453e66dd675c6882030fea547eb0eb209440f106bd4fb8ecd8bc2`；
容量SHA `81901aaeb356d343e3cf56fe9bc0caad2cb4510d63ed740ab1448c470d5d313f`。
两者160个有限LoRA张量，实际联合batch检查2次；重载旧新互换改变control/output，
source/noise相同、重复PNG一致。不是质量验证。

正式Geometry/Dynamic Binary各seed101/103、20updates/batch4/rank16/lr1e-4，从
Base独立初始化，四组训练已完成；不能与旧60-step模型宣称等预算优劣。
matched_budget_101/103均通过，两组各方法完整process/数据/容量/更新预算一致。
每个checkpoint均复核160个有限LoRA张量、实际SHA和20次联合batch验证，
peak allocated/reserved均15.0219/17.0859GiB，无失败/OOM/cleanup错误。

|seed|方法|worker(s)|含启动wall(s)|
|---:|---|---:|---:|
|101|Dynamic Binary|305.550|318.453|
|101|Geometry|306.595|319.461|
|103|Dynamic Binary|305.829|318.655|
|103|Geometry|314.845|327.940|

正式checkpoint位于 `study_01/train_<method>_<seed>/lora/train_<method>_<seed>/`
同名safetensors；实际SHA在checkpoints_verified.json：

```text
101 dynamic_binary: 65d296d5064869091c7685b7e0fa8ab054891ee70a5e2bce0c0f5bf72ba14e58
101 oracle_geometry: 910631e352e748e236fedf84cedb94860f4cb3d0ce1e32d9c3d5f35441c7fc50
103 dynamic_binary: 7011ea884c472dd98317dca70d74681c86b7e9d7c682d66cd94624d1e9f55f0e
103 oracle_geometry: 9d4df5e00efe9b0fe6611368ec28d7533dec368a100e50b89af9b4e6a0bc58ef
```

四个adapter各12个新Dev固定重载均完成，共48张；完整wall分别为Dynamic101
304.631s、Geometry101 306.763s、Dynamic103 302.878s、Geometry103 307.875s；
peak reserved均16.3945GiB。数据/adapter SHA、RGB/尺寸、prompt/seed/预算均复核。
各执行完整源码摘要固定a07ba92，无GPU执行失败或OOM，GPU资源已正常释放。

|seed|方法|非零T target MAE|D target MAE|P source drift|s=0 P drift|
|---:|---|---:|---:|---:|---:|
|101|Dynamic Binary|0.083764|0.103440|0.009316|0.013870|
|101|Geometry|0.142595|0.096412|0.008343|0.013878|
|103|Dynamic Binary|0.082599|0.102544|0.011987|0.019598|
|103|Geometry|0.143994|0.092778|0.011506|0.019508|

Geometry−Binary，scene bootstrap 2000次/统计seed42、四组场景、95% percentile：

|训练seed|T差值 [95%CI]|D差值 [95%CI]|P差值 [95%CI]|
|---:|---|---|---|
|101|+0.058832 [-0.000806, 0.118469]|-0.007029 [-0.020365, 0.000123]|-0.000973 [-0.001615, -0.000330]|
|103|+0.061396 [-0.000973, 0.123764]|-0.009766 [-0.028121, -0.000013]|-0.000481 [-0.001213, 0.000164]|

T的点估计在两seed都更差，但区间跨零，不能宣称总体显著退化；D/P的改善也保留。
这些区间仅反映固定训练seed下的场景不确定性，不代表训练seed方差，未作多重校正。
预注册的“两seed T/D改善且P/身份不恶化”必要条件未满足，**当前灰度第二参考
No-Go，不增加数据或步数**。该结论只限当前编码、12训练组、20updates；不能否定
所有角色条件方案。决策与逐项门禁见pilot_analysis_01/go_no_go.json。

预定前两组全强度联系图pilot_fixed_contacts_01中，Geometry的cube旧区出现更暗
外观，满强度cube/cylinder在两个seed仍有旧/新重复物体；Binary也重复。去掉RGB
并不等于消除VAE的外观复制。以上是非盲代理观察，尚无人工成功率或失败率。
真正独立特征投影、同容量constant/binary-feature对照继续停留设计阶段；下一次
若实施，先另预注册极小特征消费/梯度试验，不直接扩大本轮训练或进入Predictor。

## 真人与工程检查

真人使用指南已完成，旧60-step的新Dev包human_existing_01每人48任务，三份独立
顺序页面及空白CSV就绪，真实reviewer=0、评分=0、状态pending、人工区间null。
新20-step包human_pilot_01同样每人48项、三份独立顺序页面，真实评分仍0条；
两预算独立分包，干预图不进入正式盲评分。指南明确四类
失败、NA、独立性、原始CSV不可覆盖导入，工具不会邀请真人或自动填分。

Ruff check/format、214项本机Pytest/148.83s、26个CLI入口、pip check、仓库外wheel
及新配置通过。GitHub CPU CI 206 passed/8 skipped，真实GPU另行记录，不把CI称作
GPU验证。第三方与全部历史资产保留；未训练Predictor、未迁移Qwen。

最终provenance审计将17份执行记录、6份训练报告的99个源码文件摘要逐一与冻结
Git对象匹配，并复核checkpoint/config SHA、全部Dev运行与两个pending人工包，
通过；证据在final_provenance_audit.json。Stage1既有三组报告和checkpoint实际
SHA也与归档一致，未运行新的旧数据模型实验。最终实时8卡0MiB/0%、无计算进程。

当前已验证：新数据/隔离、接口消费、四旧adapter固定干预、四新adapter同预算训练
及全部Dev评估。未完成：真实三人评分/语义成功率、独立角色特征编码、广泛几何/
光照泛化。Test继续封存。历史身份大蓝块本轮未复现，仍有样本/采样相关风险。
