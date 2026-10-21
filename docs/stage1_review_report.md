# Stage 1 补充评估实际报告

更新时间：2026-10-21。工程与六组固定30例推理/像素评估已完成；180项真人盲评
全部pending。尚未建立空间Mask优于匹配无信息条件的语义结论。

## 合并与输出

Stage 1 [PR #2](https://github.com/Vincent-Wenhan/image-editing/pull/2) 在无冲突、
PR CI success后正常merge到main `a03f168`，无force push；本地130项回归及
历史78个新增blob审计通过，无禁止资产。随后从最新main建立
`feat/role-data-and-evaluation` 独立worktree，原Stage 0/1全部资产保留。

新输出根为该worktree的 `outputs/stage1_review_20261021/`。旧12例报告仍见
[stage1_report.md](stage1_report.md)，没有覆盖旧预测、指标或checkpoint。

## 对照与预处理

Constant提供全黑RGB第二参考；它经VAE编码，不是零latent。与三种Mask同为
`[source, control]`、相同VAE、T=10/20参考位置、token数与LoRA容量，使用训练器
先cast [0,1]到BF16再归一化的路径。不提供样本空间信息。

原No-mask保持单source/Diffusers预处理；新增显式
`--reference-preprocessing training` 对同一150-step No-mask checkpoint进行
cast消融。默认行为未改。真实reference smoke中，各实验初始noise相同，training
单参考与Constant双参考的source token相同；VAE输入数量与T位置符合声明，黑图
输入为全-1，training重复PNG逐字节相同。legacy/training源输入摘要确实不同。
4-step单例输出MAE=0.003547，worker77.82s；这是协议检查，不是质量比较。

30例50-step同checkpoint比较的平均输出MAE=0.002380，最大0.006917，按0.05
阈值的平均变化面积0.001433。cast有实际数值影响，不能把旧单/双参考差异全部归因
于空间信息。新的Constant与Mask路径共享预处理，避免这一比较混杂因素。

## 训练预算与真实记录

固定原200 train /30 dev、Base 4B及model/data revision、seed42、512²、
150 updates、batch4、rank/alpha16、AdamW/lr1e-4、flowmatch/sigmoid、BF16。
完整train/network/model/sample与数据处理参数通过 `budget.json` 对照原No-mask。
Constant从Base独立开始，不续训debug；三组Mask复用已完成的同预算checkpoint。
相同样本/更新预算不等于相同FLOPs，双参考耗时单列。

Constant最小2-step/batch4：worker209.92s、含启动器217.93s，peak reserved17.09GiB，
检查到2个真实双参考batch，160个有限张量。checkpoint SHA256：
`05b2017fba9da92d3ef9d89a87806dcafba85f437436ad29cc8b7b5e9268dd4a`。

Constant正式150-step：worker1153.01s、含启动器1161.16s，peak allocated/reserved
15.02/17.09GiB，双参考检查150次，160个有限张量，无OOM或cleanup错误。
checkpoint位于 `constant/lora/constant_mask/constant_mask.safetensors`，SHA256：
`78ad1bd5494ae1a6e5c778835747a1c657fd687d9df0a804318ac64fab834463`。

## 固定30例结果

使用现有全部30个dev first-turn样本；ID清单在 `fixed_samples.json`。六组同seed、
50 steps、CFG4、512²，所有实际manifest/source/target/mask SHA及条件来源复核。
统一使用原MagicBrush mask评估，不改成各方法的控制图。

| 方法 | Target MAE | Mask外漂移proxy | 变化面积 |
|---|---:|---:|---:|
| No-mask / 原预处理 | 0.058350 | 0.035196 | 0.190953 |
| No-mask / training cast | 0.058093 | 0.034881 | 0.187178 |
| Constant / 双参考 | 0.056721 | 0.033341 | 0.176775 |
| Static | 0.060950 | 0.035737 | 0.216611 |
| Dilated r16 | 0.057524 | 0.034218 | 0.184241 |
| Shuffled seed42 | 0.067604 | 0.042745 | 0.216065 |

所有组identity→target MAE=0.033798。Constant的像素误差较低，但变化面积也较小，
且有明确未执行编辑的样例；这不等于语义质量更好。Static/Dilated的像素均值没有
优于匹配双参考Constant；当前证据不足以证明空间信息的稳定收益。

| 方法 | 30图生成合计(s) | 含加载执行(s) | peak reserved(GiB) |
|---|---:|---:|---:|
| No-mask / 原预处理 | 799.62 | 885.10 | 16.26 |
| No-mask / training cast | 797.59 | 882.69 | 16.26 |
| Constant / 双参考 | 1120.58 | 1205.33 | 16.39 |
| Static | 1136.48 | 1226.18 | 16.39 |
| Dilated r16 | 1122.04 | 1203.62 | 16.39 |
| Shuffled seed42 | 1123.74 | 1204.82 | 16.39 |

`historical_png_verification.json`：原No-mask/Static/Dilated/Shuffled四组各12张、
共48张PNG，与扩大运行的对应样本逐字节相同。说明默认兼容行为保持；旧文件未改。
推理进程常驻模型，最多三组评估加独立训练并发，各自实时空闲检测与UUID租约。

## 人工与失败案例

`human_review/index.html`、`ratings.csv` 提供180个匿名任务，随机顺序固定，
source/instruction/target/prediction可见，方法映射独立保存在private目录。
评分为编辑成功度、无关内容保留0/1/2或NA，附失败类别/说明/reviewer。
导入检查运行、配对、图片及映射SHA、实际尺寸、缺项/重复、范围与评审者。
全空评分表已实际导入为180项pending；没有真人评分、成功率或人工均值。
独立数据副本按各自条件来源验证，共同展示数据另行核对，没有重写旧run.json。

以下是代理对指定案例的**非盲、探索性目视观察**，不是人工评分或总体失败率。
选择是一个历史artifact加新纳入样本中Static target-MAE最大的三例，有选择偏差。
原图与六方法联系图、选择规则和观察在 `failure_review/`。

| 样本 | 观察 |
|---|---|
| dev_319096_t1 | Static/Dilated在镜面区域出现不透明白块；No-mask/Constant可见反射图像。未确定artifact机制。 |
| dev_163114_t1 | No-mask/Constant保留摩托车；Static/Dilated替换为汽车，但外观/朝向不同于唯一reference。Shuffled有大白块。像素排名不能代替编辑成功。 |
| dev_535902_t1 | Static引入窗口结构并明显改变曝光/边框；Shuffled有白块/暗区。窗/镜完成度仍需独立评审。 |
| dev_17320_t1 | 所示方法都变蓝，但球表面/接缝明显简化；非目标属性保留需人工判断。 |

## 复现与限制

遵循README的prepare-controls/train-config/租约训练及infer入口，使用新目录、
`--limit 30`；Constant指定 `--control constant_mask` 和对应control-root。
精确argv、配置、预算、运行SHA、checkpoint、耗时均在输出中。真人评审使用
`image-edit review score` 导入下载的CSV，不能用本报告观察填充为真人评分。

五组历史checkpoint评估的controller计划保存于 `38720e9`；开发期间仅对生成调用
作模块拆分，部分排队进程未单独保存启动源码摘要，不能据controller head断言每个
worker的完整代码快照。四组前12图字节一致验证已保留；Constant另记录了启动源码
SHA。后续正式多seed研究应在冻结worktree运行并逐worker记录源码快照。

目前单训练seed、按顺序取30dev，没有正式测试集、多seed区间或真人盲评。
Mask外漂移只是proxy，MagicBrush不是物理角色真值。Stage 2仅开展最小数据和
接口工程，不把该结果视作进入Role Predictor训练的研究Go结论。

完整工程184 passed/79.97s、Ruff、依赖、editable安装与12个CLI/嵌套入口通过；
wheel在仓库外导入、配置和CLI通过。真实GPU证据仅为上述实验与独立Oracle协议
smoke，不将CPU测试当GPU验证。所有资产留本地，未上传GitHub。
