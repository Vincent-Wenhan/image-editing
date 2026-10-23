# T/D/P Role Conditioning：可学习性与失效机制预注册

2026-10-23，Asia/Shanghai。PR #6经252 tests/218.67s、55 CLI、真实96图重算、
原始SHA/配对区间及六份空白盲评包复核后，正常merge为main
8b261bb7abbcdeaa23230f6686ec472d3d6fb909（14:39:01）。新分支feat/role-learnability。
历史报告、模型、负面实验、全部资产和真实原始评分不修改；本轮真人评分仍pending。

## 问题与可识别范围

区分：少量数据上尚未学会编辑、角色差分数值过弱、注入位置不能有效利用信息。
过拟合成功只能证明固定少样本可学，不能证明泛化、物理理解或Predictor可行。
过拟合失败也不能证明T/D/P不可学习；只给当前模型/配方/预算的受限结论。
不改flowmatch/sigmoid objective、网络容量、数据真值或第三方源码。

复用既有512² Blender数据/原cohort，Train/Dev/Test不变。只选Train中按group_id
排序的前两组，每组全部s=0/0.5/1，共6对（2 identity/4非零）；不按效果筛选。
建立新的只读资产视图及caption，数据加载器只能见这6个Train target与source参考。
父dataset、canonical manifest、证据和角色保持原字节，不把Dev样本移入训练。
Dev全部12例只诊断；Test仅完整性校验，无推理/画廊/选型。

## 去除文字与随机性捷径

所有Train/Dev使用同一中性指令（唯一文本放实验YAML）；不提供位移大小、方向或
渲染参数。条件仍为显式Oracle上界old/new/D/P/valid，不是部署输入。
原始指令保留在canonical manifest，实际覆盖指令及caption SHA单独绑定。
同一scene不同强度共享source、prompt和初始噪声：seed=sample_seed(42,group_id)。
Constant三种强度应生成完全相同PNG；否则判为协议失败。Full不同强度的结果差
才可能归因于角色，不能用不同噪声制造“角色生效”。旧PNG不与新指令结果混比较。

## 有界训练与公平性

四方法Constant/Binary/Old-New/Full，独立训练seed307/311；均从相同固定Base开始，
rank/alpha16、LoRA lr1e-4、encoder lr1e-3、13,312 encoder参数、相同feature/source/
noise token数1024。复用原训练器/编码器/优化器；不从四个不同旧adapter热启动。
旧checkpoint用于冻结机制诊断，保留不恢复optimizer。

每组120updates、batch6，每update见完整6例，720次样本曝光；8组共960updates。
先单卡2-step/batch1、2-step/batch6容量与真实前反向/零条件/重载门禁，再最多4卡
正常UUID租约。OOM不自动改变正式batch/预算，失败保留并先修工程问题。
checkpoint保存0及每20步、最终120；成对encoder/LoRA均保留，禁止上游自动删旧权重。
记录每步真实loss、样本/噪声/timestep、初始参数SHA，四组同seed逐项匹配。
完成评估还逐update检查6个ID恰好各出现一次，不能仅凭batch维度认定完整样本曝光。

## 梯度、时间与层诊断

只观测原训练loss：预定update1/20/60/120记录输入各通道梯度、gradient×input、
类别/位置/输出参数梯度及LoRA梯度，有限性与幅度分列。层取实际Transformer
image projection、双流/单流各首/中/末层；记录真实梯度/特征RMS、实际timestep，
按时间分桶而不是比较不同噪声轨迹作因果归因。观测hook不能改变loss/梯度或RNG。

固定监督诊断另用t=0.1/0.5/0.9、固定group噪声，报告初始/最终teacher-forced
velocity MSE和T/D/P区域代理；target latent仅是监督和noisy target状态，绝不作
reference、prompt或Role Encoder输入，不进入生成推理。与原加权训练loss区别标明。
最终固定状态角色交换、D/P交换、空间置乱和零条件回放，比较层响应与有效信号。
完整图像比较保留真实SHA、时长、显存、位置/阴影/背景/身份和失败案例。

## 预设验收

工程通过不等于语义通过。CPU验证视图只含Train、caption/实际SHA、同组seed、
loss/梯度数值保持、hook异常清理、成对checkpoint、zero/disabled及位置对齐。
GPU完成真实训练和重载；预测不读target，teacher模块与生成模块分离。

固定Train对比：同group的预测差(pred_s-pred_0)与GT差(target_s-target_0)，报告
T/D归一化contrast MAE和有向响应（完美为0/1，Constant为1/0）。
少样本编辑拟合必要门槛：两seed Full非零T/D相对不编辑source各降低至少25%，
T MAE≤0.05、D≤0.10、P≤0.01、身份MAE≤0.01；T/D contrast MAE≤0.75。
角色利用门槛：Full对Constant及自己的zero/shuffle的T/D各改善≥5%，P/身份增量
≤0.001；D独立语义还要求对Old/New的D改善≥5%，T恶化≤5%、P增量≤0.001。
定位/阴影代理带校准与弃权，不以像素均值充当真人成功率。Train仅2场景，CI仅
描述；Dev四场景配对bootstrap2000/seed42，不据Dev调预算、挑seed或查看Test。
loss下降但角色替换无对应行为，仅算拟合，不宣称学会角色语义。

## 一个条件性最小修改

完成基线与实际梯度/层诊断后，若证据能指向一种机制，先另记唯一修改与依据，
再实现并先GPU smoke。不得同时调增益/结构/目标或做网格搜索；没有证据则不改。
最多追加同四方法×两seed×120updates的配对复验，总上限1920updates、同6例。
额外参数/token/样本均不得增加；未获预定收益停止，不进入大规模训练。

最终Go仅表示值得继续受控机制研究；No-Go限当前注入/预算。真人至少3位独立
评分仍pending，不伪造；不训练Role Predictor、不迁移Qwen，封存Test继续保留。

## 固定诊断矩阵与执行门禁

Full在update0/20/60/120保留学习曲线，其余方法在120评估。Train全部6例，
Dev全部12例；中间20/60仅Train。最终Train四方法均有original/zero/shuffle，
Full再加old/new及D/P交换。最终Dev Full取排序前两组的full-strength、首组identity
及首个非零样本做五种干预，其余只original；共356张生成图，204个原条件监督探针。
各原条件另做t=0.1/0.5/0.9监督velocity诊断。Full最终每split首个非零样本记录
真实CFG早/中/晚三个状态的分层回放及T/D/P梯度分解；不对Test做这些操作。

新诊断先在2-step容量checkpoint做单例4-step GPU smoke，核对实际轨迹回放、
zero/disabled、分层hook、监督target隔离、梯度有限性及异常清理，再运行完整矩阵。
分区梯度采用共同全图分母，记录T/D/P/ignore对全图梯度的贡献和求和误差；
这是未加权velocity诊断，不能冒充原训练器加权loss或自动微分以外的BF16导数。
机制汇总保留实际timestep、CFG分支、所选case与各层，不将梯度范数比称为可加贡献占比。

## 监督诊断预处理更正

2026-10-23：复核固定上游data_loader.py发现target路径为ToTensor/FP32 rescale，
再转换VAE dtype；source control则先cast再rescale。原teacher/gradient错误复用了
reference helper，旧数值仅属于该替代预处理，不能称为与训练target像素变换一致。
训练、实际训练梯度、生成与pixel/contrast门槛不受此问题影响，不重新训练或覆盖结果。

先新增target专用转换及覆盖全部8bit值的BF16回归，再真实GPU单例smoke。
用原checkpoint、相同Train/Dev、timestep/seed和source/text/noise，在新目录更正
全部原条件teacher与Full固定梯度探针。缓存source-only单步首状态，核对与原20-step
实际初始状态SHA；target只在状态捕获后由evaluation加载，不作为生成参考。
原记录及旧预处理的图表保留，派生更正summary单独绑定原分析和更正报告SHA。
更正不改变方法、loss、参数、正式更新预算、Go门槛或Test封存状态。
