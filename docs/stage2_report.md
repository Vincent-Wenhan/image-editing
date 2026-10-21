# Stage 2 最小反事实数据实际报告

2026-10-22增量验证见 [角色实际报告](role_validation_report.md)：已完成20个独立
train/dev/test场景组的128→512渲染及Oracle 2-step LoRA最低训练验收，正继续
同预算四方法研究。下文是2026-10-21的历史最小阶段结果，保留其全部证据与限制。

更新时间：2026-10-21。**最小数据管线、真实CPU渲染/标签校验、显式Oracle接口及
GPU条件消费已验证。未训练Oracle LoRA或Role Predictor，未迁移Qwen，没有质量
上界或泛化结论。** 设计见 [stage2_design.md](stage2_design.md)，Stage 1补充结果见
[stage1_review_report.md](stage1_review_report.md)。

## 生成器与数据

新输出根 `outputs/stage2_translation_20261021/`，原Stage 0/1全部保留。
固定官方bpy4.5.3 LTS/build `67807e1800cc`，Cycles CPU、128²、seed17、8 threads。
beauty/阴影反事实/空场为32 samples，occupancy ID为16 samples；两者的抗锯齿
覆盖并非逐像素等同，分别排除旧/新轮廓并保留背景误差门禁。禁自适应采样/降噪、
无环境光和间接/反射影响，纯Diffuse、点光源硬阴影。
全部参数在 `configs/datasets/blender_translation.yaml`，生成时保存配置与源码SHA。

4个程序化场景（cube/cylinder各两例），每场景强度0/0.5/1，共12对；train/dev各
2个完整scene group、6对，按组不跨split，source/session交集为0。同组强度共享
source，位移=s×1.5×object_width。s=0的四对source/target精确相同、标签全P。
这是同一primitive家族的不同宽度/光照组合，不能称为未见资产泛化基准。

保存source/target/coarse-source-mask、roles/valid、manifest与实际SHA；原始正常/
关闭shadow-ray/对象可见占据emission-ID/空背景EXR，以及逐对线性NPZ证据保留。
RGB按显式Standard/sRGB显示，角色计算使用线性值。

- T为旧/新可见物体占据并集，包含移走后的背景补全。
- D为T外由shadow-ray干预隔离出的阴影变化，high=0.02、low=0.002（线性RGB）。
- P为其余可信区域；旧/新各自轮廓、partial coverage、阈值间隙和未解释变化ignore。
- 标签0/1/2/255，valid为二值；全部8个非零强度样本都有可信T、D、P。

## 真实失败与修复

`render_01`完成40个EXR渲染，worker8.22s，但背景显露验证失败：
train_cylinder_b/s001两像素[55,47]、[64,47]的最大误差0.013770 >原阈值0.01。
两点都邻接新轮廓，原union外边界策略漏掉重叠内部的silhouette。

修复为分别排除旧/新轮廓；**没有提高阈值**。新增重叠边界、failed/running数据
不能验证或用于Oracle的回归。原失败目录的全部EXR/NPZ/PNG、traceback、配置和
诊断保留；常规验证/Oracle loader拒绝该失败报告。

`render_02`重新运行并通过：CPU worker8.37s，完整生成/验证12.11s，含CLI wall15.22s。
背景显露最大线性误差0.007452 <0.01。全部source/target/mask/roles/valid/evidence
与原始EXR实际SHA、split、位移、指令、共享source、数值和从证据重算的标签通过。
`render_repeat.json`确认两次运行的24张source/target PNG逐字节相同；仅有效域修复。
`role_gallery/index.html` 展示6个dev样本的source/target/mask、真实T/D/P与valid。

## Oracle 数据/模型接口

`scene_manifest.load_role_supervision` 单独读取target/roles/valid；
`load_oracle_condition` 仅检查source/labels/valid，核对canonical manifest和实际
哈希，不打开target或反事实RGB。测试删除这些RGB后仍可加载条件。
`EditInput`不增加监督字段，普通 `edit`/`infer`仍拒绝Oracle。

显式 `Flux2Backend.edit_oracle(..., allow_oracle=True)` 只接收
`[source, role_RGB]`；T/D/P为RGB单位向量，ignore黑，严格核对用途/编码/哈希/
几何。标量strength额外条件仍未实现并明确拒绝；物理强度写在指令和标签元数据。
当前不接角色训练配方；网络结构和flow loss未变，软概率契约保留但不实现预测器。

`oracle_smoke/report.json`：dev_cube_c_s001、128²、4 steps、CFG4，原Base无角色
adapter。实际VAE输入恰好两图，角色输入仅-1/+1且每像素最多一个正通道；attention
T=0/10/20各64 tokens。交换T/D后role token及输出改变，source/noise token不变；
重复Oracle PNG完全相同。worker69.85s，含启动器77.10s，peak allocated/reserved
15.19/15.22GiB，三次生成1.82/0.69/0.72s。

原/重复PNG SHA256：
`0a39e6ac1406b0032b87af60129528e07d98ddada6733aade0767ac0d6eb2560`；
置换结果SHA256：`6c38be3a81acd11110050321080ddc576202d3e88cbf57444921c40de17524c6`。
这只证明条件被消费、可重复，不能当作Oracle质量上界。

## 环境与工程检查

官方Blender二进制入口403；固定官方PyPI wheel的SHA256为
`8badbbeacce3317ca45c0963aa660f2587e2b3b707f618f9ea9af217e4049ec7`。
新软件慢链路保留收到的前缀后续传，安装日志与传输记录保留；未重新下载模型/数据。
首次导入缺libXrender等，固定9个Ubuntu 22.04包URL/版本/SHA本地提取；只在CPU
渲染子进程追加库路径、隐藏CUDA，系统/其他任务/原Conda环境不变。
显示版本有LTS后缀，按严格4.5.3数值tuple校验并保留build hash，没有放宽pin。

184项Pytest（79.97s）、Ruff、pip check、editable、所有12个CLI及嵌套/模块入口
通过；wheel的仓库外导入、配置与CLI通过。CPU fixture的EXR明确是测试占位，
不计入真实渲染；真实证据是上面的render_01/02和GPU smoke。第三方源码未修改。

## 未完成与边界

仅12对简单桌面图，直接光硬阴影，不支持软阴影、反射、间接光、多物体复杂遮挡或
真实图像的完整因果真值。有限采样及ignore边界/阈值决定标签可信范围。

尚未完成真实人工评分、Oracle LoRA同预算质量上界、多seed/更大场景与保留测试集；
没有Role Predictor、Blender大规模生成或Qwen训练。下一步先扩大scene-group并
验证Oracle能否提供语义收益，再决定是否开发预测器。所有图像/证据/权重留本地。
