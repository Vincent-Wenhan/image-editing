# 冻结角色条件：纯噪声与目标方向预注册

2026-10-23，Asia/Shanghai。PR #7最终头ff4fd8a经代码、109个text blob、
712张实际PNG、16个最终LoRA/encoder实际SHA、50份报告及No-Go门槛重算审查，
原276 tests/324.77s、67 CLI的exact-head记录及两项CI一致，Ruff重新通过。
正常merge为main 1d3986b；新分支feat/role-direction-diagnostics从该main建立。
审计在旧worktree outputs/pr7_merge_audit_20261023；历史资产和失败不修改。

## 问题与冻结范围

检验条件造成的velocity变化是否朝正确编辑方向，而非只改变输出。
仅用PR #7两个版本none/channel_rms、四方法、seed307/311的最终120-update
checkpoint，共16对；不加载optimizer，不反传、不训练、不保存/改变任何权重。
复用同一6 Train/12 Dev、512²、cohort、prompt/seed/CFG与所有原始SHA。
Train两组/Dev四组分别统计，已使用的Dev是诊断集；Test仅完整性校验，仍封存。
不训练Predictor、不迁移Qwen、不添加修复算子；无论本轮结果如何均不自动开新训练。

预算/指令/独立训练seed继承原study plan，不在新YAML复制。新配置唯一来源
configs/experiments/role_direction.yaml；只新增诊断时间、干预和数值门槛。
每组source/prompt/initial noise相同，沿用sample_seed(42, group_id)。
四方法保持原信息定义，Old/New的D/P槽是valid占位，不能误称为D/P均为零。

## 共同状态，禁止target进入模型

模型的source参考与原预处理一致。先由source-only单步pipeline捕获正/负CFG
输入、初始纯噪声epsilon与空间ID，逐项核对PR #7实际source/text/noise摘要。
另对已知source使用FP32 rescale→VAE dtype编码，得到source anchor z_s；
source参考的cast-before-rescale保持不变。anchor在target读取前建立。

主矩阵source_bridge：x_t=cast((1-t)z_s+t epsilon)，t=1/.9/.5/.1。
t=1是真实纯噪声、与原sampler初始输入相同；其他状态完全不依赖target，
是以source为端点的受控noising bridge，不冒充编辑target的训练轨迹。
辅助矩阵noise_only：同一epsilon固定，扫描t=.9/.5/.1；它是时间标签的分布外
扫描，不叫真实中/低噪声，也不用于Go门槛。t=1与主矩阵合并，不重复计证据。
每组共7个状态，强度0/.5/1、四方法和全部干预共用同一实际cast后的x_t。
使用实际BF16 timestep计算，不能用名义float替代；保留requested/actual时间。

先完成该组全部模型预测，再由evaluation读取target、角色标签并编码监督。
target latent不进入hidden states、参考、prompt或encoder；Oracle占据/D/P仍是
明确声明的上界条件，不是部署输入。预测NPZ与监督NPZ分开、不可覆盖。

## 方向定义与分数

FLUX约定v=epsilon-z_0，sampler从t大向小积分，正确编辑velocity差方向为
d_t=(z_s-z_target)/t，符号不能取反。实际端点预测为z_hat=x_t-t*v。
t=1的v_target=epsilon-z_target是精确的既有flow-matching监督方向。
t<1使用v_target=(actual_x_t-z_target)/actual_t，仅称固定状态的目标端点方向；
它不是原加权训练loss，也不是从target采样的训练轨迹，不作分布内泛化结论。

原条件v(s)与同组identity条件v(0)之差delta=v(s)-v(0)检验角色增量。
分别在T/D/P面积池化的16-pixel latent网格计算加权内积：

- cosine=<delta,d>/(||delta|| ||d||)；零能量返回null，不用0伪装相关性。
- projection=<delta,d>/||d||²；完美1，反向为负。
- contrast_gain=(2<delta,d>-||delta||²)/||d||²；完美1，Constant/零响应0。
- 分开记录absolute endpoint MSE与original相对同checkpoint zero的误差改善，
  不把LoRA通用去噪能力当作角色额外贡献。各方法raw MSE及增量都保留。

Full还与同权重drop_d比较；用D区域方向、projection及端点误差收益检验独立D，
不能仅凭Full与不同LoRA的Old/New差宣称D因果有效。Old/New Swap检验T的有向性，
D/P Swap与valid内Shuffle作预定破坏性对照；分布外干预单独不能证明语义未学会。
P报告对source的端点误差、条件额外漂移与strength contrast；VAE感受野跨区域，
latent P不等于精确背景像素，位置/阴影结论仍是代理，不能冒充真人成功率。

## CFG、BF16与空间控制

保存实际BF16 positive、negative、native CFG三个velocity数组；native公式逐算子
匹配固定Diffusers的neg+4*(pos-neg)，不先float。另算FP32 CFG仅作数值诊断，
不作为修复或改写推理。分别评估四个分支，记录正/负差分夹角、CFG抵消与舍入。
记录实际融合差分、BF16丢失、原特征/残差RMS；不以非零响应当作语义成功。
zero与disabled须逐元素等价；首状态native CFG须与真实scheduler收到的值相同。
noise/source空间ID完整唯一，按实际坐标对齐，source与text不许被干预改写。
encoder与Transformer冻结，权重文件实际SHA在前后相同，trainable parameters=0。

## 预设判断与公平统计

每个版本/seed/Train/Dev分别报告，scene内平均strength再scene等权。
mid/low固定平均t=.5/.1；high固定平均1/.9，不挑最佳时间。noise_only辅助行不入门槛。
配对bootstrap2000、seed继承42、单位scene；Train仅2/Dev4组，区间不是训练seed方差。
下面是下一轮timestep-only小试验的必要证据门槛，不是通用统计显著性标准。
native CFG和FP32 CFG都须满足，两个独立seed、Train和Dev均须通过：

1. mid/low Old/New的T，以及Full的T/D：cosine≥.10、projection≥.02、contrast_gain≥.01。
2. Old/New原条件T contrast_gain比自身Swap高≥.005；Full T比自身Swap/Shuffle、
   D比D/P Swap/Shuffle高≥.005，Full zero的contrast严格为0。
3. Full D contrast_gain比Binary和Old/New各高≥.005；T不比Old/New低超过.005。
4. Full相对同权重drop_d的D cosine≥.10、projection≥.02，端点误差减少至少
   正确编辑latent能量的.005；必须有独立D证据。
5. 四个主时间平均的Full P及identity端点误差相对zero恶化≤max(1% zero误差,1e-6)。
6. Full T/D的mid/low contrast_gain比high至少高.01，Dev配对scene区间下界>0。

若某个预定版本两seed满足全部条件，只给该版本Go“值得预注册仅改变高噪声采样
分布的公平训练对照”；本轮仍0训练。若不满足，No-Go自动推进该训练，并区分
弱/反向信号、去噪与角色脱钩、CFG数值效应、证据不足。未做采样干预，不能因
训练高噪声曝光少就断言其为主因，也不能由本次失败证明T/D/P不可学习。

## 执行与验收

CPU先测符号、完美/反向/正交/零能量、T/D/P加权、同状态、BF16数组、target隔离、
空间排列、checkpoint/config篡改拒绝、门槛及跨scene统计。复用旧CLI，增加-m研究入口。
先单卡512²真实Full smoke，测吞吐/峰值、native CFG实际回放、zero/disabled、空间
对齐和冻结；通过后最多原protocol的4个GPU任务，每次实时SMI及正常UUID租约。
保持batch1与旧推理一致，避免改变batch造成数值混杂；不为占显存重复样本。
所有输出用新目录、source/config/manifest/条件/velocity/监督实际SHA与原异常/时间/
显存记录；保存各scene表、方向热图、时间曲线和明确标记的自动失败类别。
真人包保持原始评分，真实0条就继续pending，不自动邀请或代填。

实际执行及固定门槛结果见[方向诊断报告](role_direction_report.md)。本预注册初始提交
245274f先于GPU实验，六项标准未改；原版/channel_rms均No-Go，本轮0训练。
