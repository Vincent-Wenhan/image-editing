# Proposal 评审（2026-10-20）

## 结论

建议按原路线启动：先固定 FLUX.2 Klein **Base** 4B，验证成对编辑与 LoRA，再做静态空间条件，最后检验强度相关角色预测。先不迁移 Qwen、不下载 Inter-Edit 全量集。工程路线可行；研究贡献目前是需要实验验证的假设，不能把连续控制或三角色表示本身作为新颖性主张。

## 必须补强的研究定义

1. **区分“允许变化”和“必要变化”。** [Overpainting](https://arxiv.org/abs/2609.10811) 已提出 must/may/must-not 三类控制，并用 joint attention 和 LoRA 融合图像与 trimap。我们的 D 应由明确的干预和场景机制定义，例如物体位移引起的阴影移动；不能把它改名为 may-edit 后宣称新贡献。
2. **明确强度的物理含义。** [Kontinuous Kontext](https://github.com/snap-research/kontinuouskontext) 已通过标量及指令调制编辑强度。最初只研究桌面物体沿给定方向的位移，令强度 s 对应实际距离或物体宽度的比例；扩展到材质或风格之前不要假设统一标量可识别。使用 s 表示用户强度、τ 表示扩散时间，避免都写 t。
3. **角色边界和歧义。** 建议 T 为物体旧/新可见占据区域的并集（含移走后的背景补全），D 为 T 外由干预造成的阴影/反射变化，P 为其余可信保护区。说明遮挡交界如何归类；不确定像素增加 valid/ignore 标签，避免把不确定区域强制当 P。s=0 的身份结果必须明确。
4. **单图预测存在不可辨识性。** 隐藏几何和光照不能从单图唯一恢复。预测概率角色场并报告不确定性、未见光照泛化和多解失败例，不能声称逐像素必然正确。
5. **避免标签泄漏。** 生成 predictor 输入时只允许 source、instruction、coarse mask、strength。target、精确新 footprint、物体 ID、渲染参数仅用于监督或 oracle。train/dev/test 按场景、几何资产和光照组合分组，所有强度点放在同一 split。
6. **定义反事实标签。** 固定相机、材质、曝光、采样/降噪设置并尽量共享随机数；用对象 ID 与成对无阴影/有阴影渲染隔离阴影效应。普通 RGB 差值同时包含渲染噪声和效果叠加，只能作为辅助检测。阴影/反射重叠及不可见区域必须记录有效性。

## 对照与 Go / No-Go

保留 B0–B5，但应补充 identity/no-edit、仅强度条件、与 Ours 同容量但不使用强度的 Role Predictor、固定三角色 trimap、mask dilation 半径扫描，以及可迁移的 Differential Diffusion 思路。[Differential Diffusion](https://doi.org/10.1111/cgf.70040) 的采样机制不能不经适配直接等同 FLUX rectified flow 的实现。

Static / Dilated / Soft / Predicted Role 使用同一 backbone、原始提示、训练样本和预算；固定/动态条件的编码器容量应匹配，另报推理时间和显存。Oracle 只给上界，必须单列。粗 mask 扰动要预先规定，不能针对结果调节。

正式验证至少多训练种子；bootstrap 以 scene 为单位而非把同一 scene 的强度点当独立样本。报告成功率与保护偏移的完整曲线，再给匹配完成度/变化面积的比较；单纯事后按成功样本筛选会有选择偏差。预先在 dev 定义效果大小与计算预算，测试集只用于最终结论。

先回答两件事：oracle role 在相同预算下是否比静态控制有用；predicted role 是否在未见场景上保留该收益。前者失败则排查条件接口或标签，后者失败则排查可辨识性与 predictor。通过后才考虑 Qwen 迁移。

## 工程审查与 kit 修正

- [官方模型卡](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) 确认 Base 支持多参考编辑；官方示例为 50 steps / CFG 4。固定 Base 做实验，不能在 before/after 中混入 distilled 模型。
- [官方 Edit LoRA 指南](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora) 的成对 source/target + `control_path` 与计划一致。当前训练器主分支依赖已经变化，因此固定可用 commit，不直接执行不带版本的 `pip install -U`。
- [MagicBrush 数据卡](https://huggingface.co/datasets/osunlp/MagicBrush) 定义白色 mask 为编辑区；它提供 8,807 train、528 dev、1,053 test 编辑步骤。首阶段只取 train/dev 的 first-turn 子集；按顺序取样仅用于工程 smoke，不作为无偏正式 benchmark。
- 原 kit 会对 mask 使用 Lanczos、在校验中放过缺失 mask、在预测缺失/尺寸不符时跳过、按运行顺序派生 seed，并生成未验证 YAML。新实现改为严格校验、按样本 ID 固定 seed、固定 revision、保存运行状态与显存/耗时。
- kit 的示例数据仅证明目录结构，不能代替真实 MagicBrush 运行。归档 zip 不作为可执行代码分支上传；只保留维护中的模块实现。
- 实机为 **8× RTX A6000 48GB**，并非 A6000 40GB；驱动 535.230.02。先验证单卡 BF16 和 batch，再提高有效 GPU 使用率。

## 文献核验边界

[Inter-Edit 官方仓库](https://github.com/Delong-liu-bupt/Inter-Edit) 明确发布 Qwen-Image-Edit-2511 CJT；此阶段阅读其实现并做 FLUX 上的概念适配，不声称严格复现论文。Overpainting、Kontinuous Kontext 与 FLUX/MagicBrush 的入口已核对；本评审未完成所有相关论文的逐表复现，尤其不能只凭题名断言 SliderEdit 与本方法的完整差异。
