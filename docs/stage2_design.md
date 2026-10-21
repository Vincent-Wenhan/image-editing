# Stage 1 评估补强与 Stage 2 最小反事实数据

状态：工程实现、六组30例评估、真实CPU渲染及Oracle GPU协议验收完成；真人评分、
Oracle质量上界与研究泛化尚未验证。实际结果见 [Stage 1补充报告](stage1_review_report.md)
与 [Stage 2报告](stage2_report.md)。新分支 `feat/role-data-and-evaluation`
从 main `a03f168` 创建。[PR #2](https://github.com/Vincent-Wenhan/image-editing/pull/2)
已于 2026-10-21 11:26 UTC 正常 merge；PR CI、130 项本地测试、Ruff、CLI、
依赖及历史资产审计通过，无冲突或 force push。Stage 0/1 全部 worktree/输出保留。

## 先补齐评估

- 新增 `constant_mask`：固定全黑 RGB 第二参考，所有样本相同，无空间信息；
  图序、参考数量、VAE cast、token 数、LoRA 容量与 Mask 一致。
  全黑图仍经 VAE 产生条件 latent，不能当作零 latent 或无参考。
- 固定 200 train、全部 30 dev、seed42、150 updates/batch4，沿用原配方。
  独立训练 constant，保留旧 No-mask/Static/Dilated/Shuffled checkpoint。
  不改变默认 No-mask；显式 `--reference-preprocessing training` 才用训练器 cast
  顺序，增加单参考预处理消融。双参考保持先 cast [0,1] 再归一化。
- 新评估运行全部 30 dev，同样本/seed/50 steps/CFG4；记录 ID、数据/配置/
  checkpoint SHA256、时间和显存。旧 12 例结果只读，不覆盖或冒充扩大的结果。
- 离线盲评包展示 source、instruction、target 和匿名 prediction；独立评分
  编辑成功度、无关内容保留（0/1/2），标注失败类别、简述及 reviewer。
  匿名映射独立保存，随机顺序可复现；导入核对预测哈希、任务身份、范围和缺项。
  未填评分保持 pending，像素指标不能冒充人工结果。类别含 missed_edit、
  wrong_location、collateral_change、mask_artifact、geometry、other 及无法判断。
  正式结论仍需多训练种子与独立评审。

## Stage 2 数据范围与标签

只生成程序化桌面物体水平平移，固定正交相机、材质和灯光；支持多个物理强度
`s ∈ [0,1]`，位移为 `s * max_translation_widths * object_width`，s=0 为身份。
不下载几何资产，不实现 Role Predictor、Qwen 或角色训练目标。

使用固定 Blender 4.5.3 Python 模块（官方 `bpy`，独立可选依赖）。Cycles CPU
最小渲染不初始化 CUDA；后续 GPU 渲染也须经项目租约。固定采样 seed、禁用
自适应采样和降噪、固定色彩管理；只允许直接漫反射光与硬阴影，关闭间接/
反射影响。保存正常 RGB、shadow-ray 关闭的反事实线性渲染、对象占据证据、
背景空场与场景参数，不用普通图像差分定义 D。

- T：非零强度下旧/新可见对象占据并集，包含移走后露出的背景；s=0 全 P。
- D：T 外，`(target_shadow_on - target_shadow_off) -
  (source_shadow_on - source_shadow_off)` 的最大通道绝对值超过预设高阈值。
  这是 shadow-ray 干预隔离出的效果变化，不是普通 RGB target-source 差分。
- P：T/D 外可信区域；阈值间隙、对象边界及未解释 RGB 变化区域忽略。
  标签 0=T、1=D、2=P、255=ignore；valid 单独二值记录。
- 按完整 scene group 分 train/dev，所有强度同组同 split。记录生成器版本、
  软件版本、配置/证据/图像/标签实际 SHA256；从证据重算标签，检查配对、
  强度/位移、身份、背景显露、角色、split 和相同场景来源一致性。

Oracle 显式作为上界：监督 loader 独立读取 target/roles/valid；普通 `EditInput`
不含它们。硬 T/D/P 编为 RGB 单位向量，ignore 为黑，图序 `[source, role_RGB]`。
单独 oracle 编译/后端入口要求显式标记和 provenance，默认推理仍拒绝 oracle。
当前只验收接口与条件消费，不训练 Oracle LoRA、不把未训练角色编码的输出解释
为质量上界。软 Role 沿用概率契约，不实现 predictor。

## 实施与验收

1. 先提交设计和 README/架构/实验日志。
2. constant、显式 cast 对照、固定样本/盲评导出导入及回归；GPU smoke 后正式预算。
3. Blender 场景/证据生成、标签、manifest 与只读校验，真实小规模 CPU 渲染。
4. Oracle 数据/模型接口，测试 target 无泄漏、通道/值域/几何及默认拒绝。
5. Ruff/Pytest/CLI/安装/资产检查；记录真实耗时、失败、checkpoint 和限制。
   按 Conventional Commits 分阶段提交并正常 push。

新输出根为 `outputs/stage1_review_20261021/` 与 `outputs/stage2_translation_20261021/`，
重复尝试必须换子目录。人工评分在 reviewer 实际填写前保持 pending；小型程序化
数据只验证生成和标注机制，不代表泛化。

参考：[Blender 官方 Python 包](https://pypi.org/project/bpy/4.5.3/)、
[Cycles 对象可见性](https://docs.blender.org/manual/en/4.5/render/cycles/object_settings/object_data.html)、
[Inter-Edit 走读](../notes/inter_edit_code_review.md)、[proposal 审查](proposal_review.md)。

### 无系统写权限的渲染依赖

服务器首次导入bpy缺少X/GL动态库；补充固定Ubuntu 22.04包URL/版本/SHA256的本地
运行库引导，仅提取到忽略目录并在Blender子进程的LD_LIBRARY_PATH使用，不改变
命名Conda环境或其他任务的动态库。普通已安装系统库的环境无需此路径。

### 角色核验视图

复用现有gallery入口增加显式show-roles选项；验证完整场景后展示source/target/
source-mask、T/D/P RGB和valid，不改变旧布局。角色图必须由实际标签/有效域生成，
用于本地核验，不能当作模型预测或研究质量结论。
