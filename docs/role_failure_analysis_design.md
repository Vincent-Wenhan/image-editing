# Role Conditioning Failure Analysis

2026-10-22 UTC，起点 main `a647522`（PR #3 正常 merge；最新 head 两项 CI 通过）。
前轮正式归档为 Causal Role Data and Oracle Feasibility Study；彩色 Oracle No-Go
和全部资产原样保留。本轮不训练 Predictor、不迁移 Qwen、不修改第三方源码。

## 预注册范围与门禁

新目录 `outputs/role_failure_20261022/`。新采样 seed=2026102201，512²，12 train /
4 Dev /4封存 Test groups，强度0/0.5/1；与旧研究所有 split 比对真实 group/source
SHA，禁止重叠。Test 仅生成、完整性校验、记录摘要，不做模型推理、画廊、评分或
选择。本轮所有模型实验仅用新 Dev；旧 checkpoint 权重只读重载。

固定前两个新 Dev groups 的全部强度用于干预，不按模型效果挑样本；完整12例用于
原条件及训练试验评估。旧 Oracle/Dynamic Binary 的 seed42/43 checkpoint 均保留。
采样20steps、CFG4、ID seed42、512²；同一样本的干预保持 source/noise/prompt/
adapter/参考数不变，保存实际控制PNG、VAE/token摘要和输出SHA。干预属于现有
checkpoint 的分布外敏感性诊断，不能当成重新训练的公平质量比较。

- Oracle：原one-hot、循环RGB配色（空间标签不变）、P改黑（T/D不变）。比较输出
  色度随配色变化、全P身份漂移，不能仅凭颜色相关就证明完整因果机制。
- Dynamic Binary：原T∪D、仅旧占据、仅新占据。单独检查旧物体残留/重复和目标
  位置，像素区域误差不是物体计数/语义成功率。干预中删除D等信息须明确标注。
- T=旧∪新在交换旧新时完全相同，属于编码不可识别性；用真实占据验证，不把
  同一控制图送两遍得到相同输出称为模型理解了方向。

## Role-aware Conditioning V2 备选

|方案|机制|代价/风险|本轮状态|
|---|---|---|---|
|独立空间角色特征编码|角色/valid直接patch化，经独立投影到Transformer特征；不走RGB VAE；旧/新各通道，P默认零残差|新增参数、训练/推理特征与位置对齐、checkpoint版本及零初始化门禁；需匹配容量消融|设计，不实施/不训练|
|独立旧/新二值参考|三参考 source/old/new，另处理D|可复用上游多图，但增加tokens且仍会复制外观；必须三参考无信息对照|设计，不实施|
|灰度几何第二参考|同双参考/同VAE；P为0，旧独占64、新独占192、重叠128、D255；valid外0|无彩色图但VAE仍可能复制灰度形状；值间距离不是物理度量|最小可执行试验|

最小V2命名 `oracle_geometry`，仅显式真值上界。它分离旧/新占据而不加载target RGB；
推理只取 NPZ 两个占据数组、角色/valid，验证实际SHA；反事实RGB只在数据校验/
训练监督/评估中使用。身份强度0全部条件为黑；不把原角色P的蓝色平面送入VAE。
调换旧/新必须改变几何编码，T∪D并集保持。它同时改变P和位置编码，无法单独归因
全部变化；独立消融需后续预注册，不能以本轮小样本消除该风险。

独立特征方案的后续接口建议：输入同尺寸 old/new/D/P 概率与 valid，按后端真实
patch网格池化为 `[B,N,C]` 空间特征，valid单独门控，不把ignore当P。在FLUX后端
用独立小投影生成角色tokens/残差及明确类型/位置标记，source仍走现有图像VAE；
训练和采样共享完全相同的编码、网格和权重。这样角色不再经过自然图像RGB重建
空间；是否减少复制只是待验证假设。初始残差为零、注入强度有界，先验证零输入
复现已有路径、角色置换/旧新交换的条件敏感性和有限梯度，再做小规模训练。
old/new是可重叠的占据概率，不强制其和为1；若采用互斥分类，可改为old-only/
new-only/overlap/D/P，valid仍单独保存。原T/D/P软场的归一化契约不因此改变。

独立encoder增加参数，必须与同容量的constant-feature、binary-feature对照保持
token数/LoRA/优化预算一致；encoder与LoRA共同保存，记录其独立SHA和编码版本，
缺失encoder时明确拒绝加载，不能用零特征静默退化。旧RGB checkpoint不能被称为
已适配该特征空间。Qwen将来单独适配其网格/Transformer，不强行共享内部结构。
本轮没有实现这些层、保存格式或训练，也不因灰度pilot完成就默认推进。

旧/新并集对调不变只说明**控制图本身**没有方向标记；完整输入还有source和指令，
不能据此声称整个编辑任务不可辨识，或认定重复物体必然由此引起。灰度方案也是
有序数值而非真正独立的角色特征，仍有VAE外观复制和类别距离偏置风险。

同数据/同预算重新训练 Dynamic Binary 与 Geometry，独立训练seed101/103，
各20updates/batch4/rank16/lr1e-4，从Base独立初始化，保持原flow loss。旧60-step
adapter仅诊断，不与20-step结果声称等预算优劣。先Geometry2-step/batch1及真实
重载干预smoke，实测后再batch4和最多四卡并行，不自动扩大研究batch。

进入下一轮的必要条件：两个seed在新Dev非零T/D均改善、P不恶化；身份误差不增加；
无明显复制/重复/目标位置退化。真人评审未完成时最多得到provisional结果，不能
宣布语义通过或扩大训练。失败/不确定则停在小试验，保留checkpoint与全部失败。

## 真人使用准备与验收

沿用离线盲评、不可覆盖CSV导入与至少三人配对区间。新Dev包独立于旧评审包；
明确重复物体、错误目标位置、颜色复制、身份破坏的判断规则。占位代号不代表真人，
实际收到之前0条/pending/null，禁止代理填写或自动生成任何评分。

验收含数据/编码/权限/无target泄漏、真实SHA、旧新互换、身份与已实现原协议回归；
Ruff/Pytest/CLI/安装检查及真实GPU smoke。每个worker冻结源码、版本、配置、预算、
seed、GPU UUID、checkpoint、实际时间/峰值/失败。所有新输出禁止覆盖。
