# 唯一最小修改：角色通道空间 RMS 预条件化

2026-10-23，Asia/Shanghai；实现与新训练前预注册。只验证一个输入预条件算子，
沿用原Role Encoder、FLUX.2、LoRA与flow-matching目标，不增加网络或参数。
全部原数据、失败目录、checkpoint、预测和原条件结果保留。

## 依据与边界

study_03/seed307四组120updates实际初始化、样本、噪声、时间步与容量逐项匹配。
原Full类别/位置输入RMS比仅0.0773，update120 categories梯度RMS：
Old5.668e-7、New4.983e-7、D3.380e-7、P1.625e-5、valid1.621e-5。
四组first20/last20 loss相近，Full为0.036867/0.024777；不能仅据下降宣称角色可学。
固定Train6例全部生成后，非零T/D contrast MAE为0.999039/1.002119，有向响应
0.001377/-0.002302；Constant为1/1、0/0。Full T/D MAE0.168249/0.311111，
source参考0.168721/0.333698，角色置乱和Old/New交换未显示正确位置控制。
证据：preliminary_seed307.json、preliminary_train307_metrics.json及其原始预测。

最终Full首个非零Train案例的Old/New差分在实际BF16加法中有79.8–84.1%丢失，
仍产生1.75–7.81%的CFG速度RMS差；梯度有限且非零。t=0.1/0.5/0.9的P区域
梯度范数比为0.970/0.846/0.589，D为0.056/0.167/0.243。范数不可加，不是贡献占比。
这些证据支持“低覆盖角色的有效信号弱”这个可检验假设，不能排除注入位置、
全图目标、少样本预算或语义未学会；更不证明单个原因足以解释全部失败。
候选只依据已完成的Train观测及既有诊断提出，不用新Test或挑Dev案例调参。
原双seed/全部Dev矩阵继续完成；未预设修复有效。

## 算子与兼容性

设原始pooled字段F为BxNx5、范围[0,1]，类别输入改为F/rms_N(F)，逐样本逐通道。
RMS统计stop-gradient；RMS≤1e-6的空通道用分母1，不放大不存在的角色。
仅改变类别线性层前的数值尺度，位置编码、隐藏层、输出tanh/max_residual和原valid
输出门控不变。常数全1条件严格不变。角色图、valid、source、噪声和坐标不改变；
归一化不访问target或GT RGB，不重新定义角色真值或flow loss。

新增显式encoder配置normalization=channel_rms；默认none沿用原运算，旧metadata
可重载，旧None/Mask路径不启用算子。新权重metadata必须记录归一化选项；不在
原checkpoint上仅改变推理算子当作新训练结果，不试增益、位置或loss的其他改法。

## 单次有界对照与门槛

CPU先核对空通道/有限梯度、原始字段不变、constant及零初始化等价、容量与旧
checkpoint兼容。真实GPU两步batch1、前反向/置换/空间/zero/重载，再batch6容量。
随后同四方法×seed307/311各120updates/batch6，只使用相同6 Train例、中性指令和
全部原优化/采样预算。所有组均应用同一预条件规则；每组仍13,312 encoder参数、
23,114,752总训练参数、1024 feature/source/noise tokens。正式追加最多960updates，
与原实验合计1920；debug更新另列。不搜索或延长预算。

使用新study、独立view/缓存、输出和成对checkpoint。实际初始权重、每步ID/噪声/
timestep、数据/配置/权重/预测SHA及源码完整记录。全部356图矩阵与原门槛复用，
Constant原/新对照检查实际参数和预测，不能将额外训练或数据归因于角色。

继续投入的数值必要条件：两seed Full相对各自原版本的非零T/D MAE各改善≥5%，
P/identity增量≤0.001；还必须通过原少样本fit、zero/shuffle/Constant角色利用及
相对Old/New的D增量门槛。单有梯度或残差放大不算通过，loss下降不算角色利用。
Dev仅诊断/场景配对CI，Test封存，真人评分仍pending；过拟合成功不代表泛化。
若未通过则停止本轮改法与扩量，不再追加第二个结构或超参尝试。

诊断勘误：随后确认原teacher/监督梯度用了source的cast顺序，上述监督梯度范数
属于该旧代理预处理，不能称为训练target转换一致。实际训练参数梯度、实际像素/
contrast、BF16轨迹回放不受影响；候选、预算与门槛不改，另存更正监督诊断。
