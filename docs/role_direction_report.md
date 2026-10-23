# T/D/P 纯噪声条件方向诊断

2026-10-23，Asia/Shanghai。范围是冻结权重的方向诊断，不是新训练或新模型。
预注册见 [role_direction_design.md](role_direction_design.md)。

## 合并与执行边界

PR #7头ff4fd8a经代码、最终CI、实际PNG/成对权重/报告SHA及原No-Go门槛复核，
正常merge为main 1d3986b；没有force push。审计存于旧可学习性worktree的
`outputs/pr7_merge_audit_20261023/audit.json`。新分支feat/role-direction-diagnostics
从该main创建；合并后main CI39956819220成功。历史112对checkpoint、712张图、
失败日志和评分包均保留。

GPU可执行源码冻结444ac6b，CPU评分/作图源码冻结daa8bf7，分别使用独立、干净的
detached worktree，editable安装复用现有环境/资产，不重新下载。
新资产统一存于本分支worktree的`outputs/role_direction_20261023/`，不进Git。
本轮训练更新0，不加载optimizer、不反传、不修改模型权重；Test继续封存。

## 样本与度量

复用PR #7 original/channel_rms的Constant、Binary、Old/New、Full Role，
两独立训练seed307/311的最终120-step成对checkpoint，共16组。
固定6 Train（2 scene groups）、12 Dev（4 scene groups），512²；prompt、source、
sample seed42及CFG4均继承原计划。两版本同数据/参数容量/原训练预算，未增加曝光。
Old/New的D/P槽是valid占位；Binary包含Oracle变化区域，不能当成无信息输入。
不同方法的已训练LoRA权重不同；跨方法胜负不能单独归因于某个通道，因此另保留
同checkpoint的strength、zero及dropD对照。

每scene采用相同epsilon、source anchor和空间ID，主矩阵t=1/.9/.5/.1；
t=1为真实初始纯噪声，其他t是target-free source bridge，使用实际BF16时间。
同一epsilon的noise-only低时间标签扫描是分布外辅助诊断，不参与门槛。
每组126个case-state，六干预original/zero/shuffle/old-new swap/D-P swap/dropD。
先完成预测，再独立编码target及T/D/P区域；target不进入模型参考或hidden states。

FLUX约定v=epsilon-z0、积分dt<0。正确编辑velocity增量是
`d=(z_source-z_target)/actual_t`；role contrast是同状态`v(strength)-v(0)`。
报告cosine、投影及`contrast_gain=(2<delta,d>-||delta||²)/||d||²`，零能量返回null。
通用端点误差、相对同checkpoint zero的额外收益和同权重dropD的独立D效应分开。
t=1目标方向是精确FM监督；其他t只称固定状态端点代理，不冒充上游训练loss。
T/D/P在16-pixel面积池化网格上加权，VAE跨区域耦合使latent P不等于像素保持。

保留positive、negative、逐算子BF16 native CFG及仅供诊断的FP32 CFG。
Go门槛要求两个CFG、两seed、Train/Dev都通过，禁止挑time或Dev再调门槛。
scene内平均strength、scene等权；2000次paired scene bootstrap、seed42。
Train仅2/Dev4组，区间不估计训练seed或广泛场景泛化不确定性。

## 已运行的工程验收

- 真实4B/512² smoke：186.54s，allocated/reserved峰值15.595/16.268 GiB，
  14 case-state；native CFG与实际scheduler输入一致，zero/disabled等价，空间对齐通过。
- 完整16/16组GPU完成，无失败；单组185.00–590.02s，累计6334.82 GPU-worker秒，
  峰值allocated/reserved 15.614/16.268 GiB。累计worker秒不等于并行墙钟。
- 每次通过实时SMI空闲检查和正常UUID租约；使用GPU4–7，GPU0–3他人进程未触碰。
  单卡smoke后最多4并发、模型常驻、batch1保持数值可比，结束后4–7均释放。
- 303 passed/222.52s（完整本机依赖，CUDA隐藏）；73 CLI通过/110.01s；
  Ruff及最终format 225文件、editable安装、pip check、仓库卫生检查通过。
- 代码CI39966675465成功：264 passed/39 skipped（远端没有Torch），
  不将其代替真实GPU验证。本地包含安装后仓库外导入、CLI及既有兼容性回归。
- ai-toolkit a055947、Inter-Edit 220c1dc、Diffusers d791c5c固定且tracked源码干净。
  Torch2.6.0+cu118、Diffusers0.37.0、Transformers4.57.3、PEFT0.18.1。
- 历史6份真人panel实际SHA未变，288行仍空白，0评审者/0真实评分，pending。

完整GPU编排墙钟1987.24s，smoke含启动228.85s；独立CPU分析412.55s、画廊72.96s。
`analysis_01/summary.json`保存48,384条分支分数、12,096条机制观测及全部配对区间。
6216个去重BF16预测归档的实际SHA、原生CFG、实际融合、空间坐标/角色干预均经
冻结分析重验；288个监督归档独立保存，跨16组source/text/noise/anchor和监督一致。
32个最终权重文件再次实际SHA检查，GPU/CPU源码文件分别与Git提交逐个核对。
`experiment_audit_01.json`绑定这些结果，`gallery_01/`含8套曲线、24个全scene联系图，
各PNG/SVG共64文件且实际SHA重验。233条自动case-time风险行不是人工失败率。

## 实际方向结果

**结论：原版与channel_rms全部16个family/seed/CFG/split切片均未通过预设门槛。
No-Go仅调整高噪声采样分布的下一轮训练，本轮未启动任何训练。**

下表均为Dev、原条件、native CFG的scene等权均值。gain为原生分数，完美1，
Constant的strength contrast严格0；负数表示条件增量没有缩短正确编辑向量的误差。
mid/low固定平均.5/.1；全Train、所有干预/四分支/时间及逐scene原值都在summary/CSV。

| 版本 | seed | 方法 | T gain t=1 | D gain t=1 | T gain mid/low | D gain mid/low |
|---|---:|---|---:|---:|---:|---:|
| original | 307 | constant | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| original | 307 | binary | +0.002376 | +0.001601 | -0.000230 | -0.000322 |
| original | 307 | old_new | -0.000586 | -0.001010 | -0.000439 | -0.000813 |
| original | 307 | full_role | -0.002009 | -0.000172 | -0.000656 | -0.000741 |
| original | 311 | constant | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| original | 311 | binary | +0.005543 | +0.003772 | -0.000525 | -0.000180 |
| original | 311 | old_new | -0.002700 | -0.001382 | -0.000944 | -0.000323 |
| original | 311 | full_role | +0.000499 | +0.001695 | -0.000917 | -0.000328 |
| channel_rms | 307 | constant | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| channel_rms | 307 | binary | -0.008258 | -0.005055 | -0.000331 | -0.002301 |
| channel_rms | 307 | old_new | +0.005579 | +0.000427 | -0.000274 | -0.000040 |
| channel_rms | 307 | full_role | -0.009889 | -0.001525 | -0.000794 | -0.000568 |
| channel_rms | 311 | constant | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| channel_rms | 311 | binary | -0.005202 | -0.002132 | -0.000269 | -0.001034 |
| channel_rms | 311 | old_new | +0.000782 | +0.002501 | -0.001069 | -0.000679 |
| channel_rms | 311 | full_role | -0.011377 | -0.008300 | -0.000464 | -0.002942 |

在真正纯噪声t=1处，Full的T方向cosine：原版seed307/311为−.019890/+.017618，
channel_rms为−.130104/−.146357；对应投影−.000510/+.000722、−.004357/−.005148。
Full相对Binary的T gain两版本两seed均为负，但4个scene的区间多数跨0，不声称
在更广场景普遍劣于Binary。Full−Old/New的channel_rms T gain为−.015468
[-.020821,−.008128]、−.012159 [−.019088,−.005815]；这是固定Dev/seed内的负证据。
原版Full的D相对Old/New在t=1有小正差（+.000838/+.003076，区间均跨0），
不足以推断D语义：独立消融、跨seed及预定时间条件仍需通过。

### Old/New和独立D

Old/New在channel_rms t=1出现小的正T信号（cosine .096138/.031905，
投影.003342/.000892），但mid/low投影仅+.000251/−.000199，未达到.02；
原版mid/low为+.000118/−.000156。Old/New原条件相对Swap的mid/low T gain
差为−.000244、−.000517、+.001238、−.000348（原版307/311、RMS307/311），
没有稳定有向性，不能把单点正相关当成学会旧→新位移。

Full同权重dropD的Dev mid/low结果如下；这比跨LoRA的Full−Old/New更接近独立D检验。
error gain按正确D编辑latent能量归一化，预设门槛分别为.10/.02/.005。

| 版本 | seed | D cosine | D projection | D error gain vs dropD |
|---|---:|---:|---:|---:|
| original | 307 | -0.004007 | -0.000077 | -0.000234 |
| original | 311 | -0.000218 | +0.000026 | +0.000047 |
| channel_rms | 307 | -0.014701 | -0.000387 | -0.000877 |
| channel_rms | 311 | -0.059826 | -0.001300 | -0.002607 |

Full D原条件相对D/P Swap的mid/low contrast差依次−.000235、+.000006、
−.000144、−.002328，均不足.005。RMS311的positive/negative D cosine
分别−.126272/−.133073，方向错误已出现在两个单独分支；不能仅归咎CFG。
D消融是上界特征的分布外干预，负结果不等于证明D不可学习；本次没有可靠的独立阴影控制证据。

### 去噪、P保持与数值机制

absolute endpoint MSE、相对zero的conditional gain和strength contrast不能互换。
例如RMS311 Full在t=1相对zero的T/D端点误差改善+1.84%/+1.29%，但其T/D
strength contrast gain为−.011377/−.008300：改善通用去噪不代表完成角色指定编辑。
这也解释了为何没有采用单独的输出差异、非零梯度或raw MSE胜负作为语义成功标准。

Full四个主时间平均的Dev P误差相对zero恶化：原版307/311为3.51%/3.87%，
RMS为3.51%/2.85%；identity全P条件分别3.49%/4.10%、3.57%/2.68%，
均超过预设1%容许值。Train也未通过。latent P是保持代理，不直接等同像素或真人身份评分。

描述性地汇总两个训练seed、四个主时间、非零strength的Dev条件交换：
Old/New实际BF16融合中，交换产生的非零FP32残差差分有83.09%在原版被舍入丢失，
RMS降至32.84%；Full为83.23%→30.85%。这说明RMS让更多角色差分进入网络，
但并未使其沿正确目标方向。这里只描述数值，不把跨seed均值当成新的正式检验。

Full mid/low native与FP32 CFG的T/D gain均值差均小于1e-5，两者门槛结论完全一致。
正负分支的strength差分夹角均值接近正交而非系统性反向抵消，且单独分支也未形成
稳定正确方向。FP32 CFG只移除guidance末端舍入，**不是**FP32 Transformer/融合，
故不能排除所有BF16效应；数值改善与非零信号本身不是语义有效的充分证据。

### 时间差、场景与失败可视化

预设Full mid/low−high gain差要求≥.01，Dev CI下界>0；以下原生CFG结果均未通过。

| 版本 | seed | T gap [95% CI] | D gap [95% CI] |
|---|---:|---|---|
| original | 307 | +0.000561 [-0.002945, +0.002779] | -0.000583 [-0.002702, +0.001536] |
| original | 311 | -0.000461 [-0.005018, +0.004097] | +0.000254 [-0.001407, +0.003216] |
| channel_rms | 307 | +0.005211 [+0.001115, +0.009307] | +0.001099 [-0.000184, +0.002381] |
| channel_rms | 311 | +0.005419 [+0.003246, +0.007591] | +0.001920 [-0.000928, +0.004768] |

RMS的T差虽在Dev为正，但主要是高噪声更差，mid/low角色方向本身仍接近0/错误，
D、P及跨Train/seed必要证据均缺失。不能凭这个差异启动高噪声采样训练。
固定epsilon的noise-only辅助扫描也已保留，属于分布外time标签干预，不作为训练曝光证据。

24个联系图覆盖全部6个Train/Dev scene groups、两seed和两个版本的Full strength1，
用同一色阶比较四个主时间的有向dot和D端点gain，并旁列历史PR #7的20-step图。
另对按字典序首个Dev场景的四组图做非盲agent检查，存`inspection_01.json`，不是人评：
原版307/RMS307的历史物体偏到旧位置下方，原版311偏上且长斜影，RMS311虽向右
仍偏离目标高度并改变朝向/阴影；热图呈混合/反向局部响应，未给出一致目标位移证据。
这些是已有图像的失败案例注解，不是本轮新生成质量实验或成功率。

## 解释与下一步

可以确认的失效表现是：条件进入网络后，额外velocity多为弱、近正交或反向；
Full没有稳定超过Binary/Old-New的正确增量，D没有跨条件/seed的独立效应，P仍被破坏。
因此不是“只要信号非零就有效”，也不能用RMS数值改善代替任务方向改善。

本实验**不能因果区分训练曝光不足、条件表示/注入位置限制和领域拟合不足**。
没有改变训练采样分布；source bridge的低时间也不是target训练轨迹，Dev仅4组。
部分t=1负向可以直接验证，但不能由此证明T/D/P在其他机制或充足训练下一定不可学。
既有训练loss/曝光与失败记录见PR #7报告，本轮没有新的训练loss。

当前建议保留T/D/P反事实数据、区域上界及诊断框架，暂停用现有注入配方扩大投入。
优先完成真实盲评，核验现有位移/阴影任务与失败类型；之后如另立受控研究，先获得
两seed下正确条件优于交换/置乱、独立D和P保持的正面方向证据，再考虑仅timestep
采样的公平训练对照。该建议不授权新模型、额外修复、Role Predictor或Qwen；本轮均未实现。

## 资产定位与复现

- 数据沿用`image-editing-role-failure/outputs/role_failure_20261022/scenes512_01`；
  Train manifest SHA `690d3c864409c5d563c69652ef3e21946e119fec60d467ddc6d1c20862e2f8f9`，
  Dev `b6778f19688023a6720041fe9f1a8b69fcd69e4b2d505b32784ee3a020efb29f`，
  Test `3a6f26bc9bb4896066046a5597d5fa7bea59c94ecfdbadc3f5decda1c6c8d68b`（仅完整性）。
- 两个父study为`image-editing-role-learnability/outputs/role_learnability_20261023/`
  下`study_03`、`normalization_study_01`。每组最终LoRA为
  `train_<method>_<seed>/lora/train_<method>_<seed>/train_<method>_<seed>.safetensors`，
  encoder同名`.features.safetensors`，update120，13,312个encoder参数；
  全部绝对路径/实际SHA/metadata在新`study_01/plan.json`，无新权重。
- 模型固定未蒸馏FLUX.2 Klein Base 4B revision a3b4f4849157f664bdbc776fd7453c2783562f4d。
  六干预/时间/门槛唯一配置`configs/experiments/role_direction.yaml`，原预算不复制。
- 新`study_01/{smoke,diagnose}_phase_01.json`、每job `report.json`及相邻
  `_execution/{execution.json,process.log}`保留正常租约、版本、耗时、显存、原异常位置。
  本轮无GPU失败，不删除以往失败；GPU前后inventory单独保存。
- `analysis_01/{summary.json,metrics.csv}`含完整逐例/逐scene可重算分数与配对区间；
  `gallery_01/{index.html,index.json,automated_failures.csv}`索引曲线、热图和自动风险行。
  运行命令见README，CPU输出/图像须新目录，画廊需可选`.[plots]`依赖。
- `experiment_audit_01.json`、`external_audit_01.json`、`human_preservation_01.json`及
  最终验收receipt分别绑定实验、第三方、真人原始表及Git/CI。未收到真实评分，未推断人类成功率。

固定资产摘要（SHA256）：

| 资产 | 实际摘要 |
|---|---|
| 新study plan | `da2a82c894b31ac23df9783eeeaca40c2adad9922cbc3cf4b4fc53db0e3556a6` |
| 完整方向summary | `e2b413d867b3c1a2885434d06f184f5da3fc08df4cab3ef72371ba22878f967b` |
| 实验审计 | `9f119d5a3c0bdf75c3feb6753b90367b9b14ce9d9615892d286ceca00fa8c5ef` |
| 图表索引 | `adcc29759fde8cb958474d3316200831f9efba08589371b20a340659685cd2ea` |
