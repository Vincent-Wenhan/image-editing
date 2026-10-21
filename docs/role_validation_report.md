# 角色研究验证实际报告

更新时间：2026-10-22（Asia/Shanghai）。本轮新输出根
`/home/jiangwenhan/project/image-editing-role-data/outputs/role_validation_20261022/`。
全部历史资产保留，图像/数据/模型/checkpoint不入Git。设计与预定预算见
[role_validation_design.md](role_validation_design.md)。

## 已验证的数据与评估工具

20个固定seed=20261022的物理scene groups，12 train /4 dev /4 test；每组强度
0/0.5/1，共36/12/12对，光源、对象位置、宽度、颜色和背景棋盘程序化变化。
同一组所有强度留在一个split；所有split两两session/source哈希无交集。
仍是cube/cylinder家族、直接光硬阴影，不是未见几何家族或复杂物理泛化。

128² `scenes128_01`完整生成/验证48.4411s，背景显露线性误差max0.00630373。
512² `scenes512_01`相同物理组完整生成/验证473.4204s，max0.00743136。
均低于原0.01门禁；未放宽线性role阈值和轮廓ignore。配置/EXR/NPZ/PNG、全部
反事实标签重算、有效域、位移与身份样本通过。新role/valid画廊在 `scene_gallery/`。

Stage 1三组150-step历史训练与30例重载公平性/分组复核已完成，见
[Stage 1补充报告](stage1_review_report.md)。三位独立评审的导入/合并/配对区间
代码和fixture测试通过；实际真人0位、评分0条，全部pending，没有伪造人工结果。

## 执行版本与最低GPU验收

实际GPU代码冻结在独立worktree
`/home/jiangwenhan/project/image-editing-role-validation-run`，commit `6879c87`。
每个subprocess的execution.json与训练worker均保存源码SHA、argv、config、依赖、
实际GPU UUID与时间。后续CI fixture修复不改变冻结的生产代码。FLUX.2 Base与
ai-toolkit/VAE/TE均复用固定本地资产，第三方未修改，没有重新下载模型。

`study_01/smoke_oracle_role_42`：512²、2 steps、batch1、rank/alpha16，worker
143.7247s，peak allocated/reserved15.0219/15.0371GiB；真实one-hot双参考batch
检查2次，checkpoint有限性检查通过。checkpoint：
`smoke_oracle_role_42/lora/smoke_oracle_role_42/smoke_oracle_role_42.safetensors`，
SHA256 `7fd99d95c42ffc6a069cb9e98a9bab4c02c59a66f80f9b36afbe4211df947ac6`。
这仅是训练工程smoke，不能称为质量上界。

重载条件敏感性、batch4容量及四方法双seed60步研究当前在执行/准备中，结果pending；
最终完成前不宣称通过。所有方法仍使用原flowmatch/sigmoid目标，没有新增loss。
Dynamic Binary=valid内T∪D，也依赖真值，与Oracle一起仅走显式上界入口。

## 检查与当前边界

本机199 passed/88.62s、Ruff check/format与CLI通过。首次新代码GitHub CPU CI
因模拟backend测试遗漏包版本stub而失败（190 passed/7 skipped/2 failed），已修复；
`fe0029f`的push和PR checks均success。真实GPU报告与CPU fixture分开。

Go/No-Go尚待已计划的同预算实验；目前只足以继续上界测量，不支持进入Role
Predictor训练或Qwen迁移。人工评审仍需至少3位真实独立评分，不能由代理代填。
